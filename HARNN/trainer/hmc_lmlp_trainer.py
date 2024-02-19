import copy, datetime
import torch
import sys, os
import numpy as np
sys.path.append('../')
from utils import data_helpers as dh
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit,MultilabelStratifiedKFold

from torchmetrics import AUROC, AveragePrecision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from HARNN.model.hmc_lmlp_model import activate_learning_level

def get_local_class_range(num_classes_list,level):
    begin = 0
    end = num_classes_list[0]
    for i in range(level+1):
        if i == 0:
            begin = 0
            end = num_classes_list[0]
        else:
            begin += num_classes_list[i-1]
            end += num_classes_list[i]
    return begin, end

class HmcLMLPTrainer():
    def __init__(self,model,criterion,optimizer,scheduler,training_datasets,num_classes_list,path_to_model,pcp_hierarchy,args,device=None):
        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = device
        self.pcp_hierarchy = pcp_hierarchy
        self.best_model = copy.deepcopy(model)
        self.args = args
        self.num_classes_list = num_classes_list
        self.path_to_model = path_to_model        
        self.tb_writer = SummaryWriter(path_to_model)
        sharing_strategy = "file_system"
        def set_worker_sharing_strategy(worker_id: int):
            torch.multiprocessing.set_sharing_strategy(sharing_strategy)
        # Create Dataloader for Training and Validation Dataset
        kwargs = {'num_workers': args.num_workers_dataloader, 'pin_memory': args.pin_memory} if self.args.gpu else {}
        self.data_loaders = []
        for i in range(len(training_datasets)):
            self.data_loaders.append(DataLoader(training_datasets[i],batch_size=args.batch_size,shuffle=True,worker_init_fn=set_worker_sharing_strategy,**kwargs))  
        
        
    def train_and_validate(self):        
        for level in range(len(self.num_classes_list)):
            counter = 0
            best_epoch = 0
            best_vloss = 1_000_000.
            # Generate one MultiLabelStratifiedShuffleSplit for normal Training.
            train_loader,val_loader = None,None
            
            # Start with best_model in every level.
            self.model = copy.deepcopy(self.best_model)
            msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
            X = [image_tuple[0] for image_tuple in self.data_loaders[level].dataset.image_label_tuple_list] 
            y = np.stack([image_tuple[1].numpy() for image_tuple in self.data_loaders[level].dataset.image_label_tuple_list])
            # Reset learning rate if new level is learned.
            self.optimizer.learning_rate = self.args.learning_rate
            for train_index, val_index in msss.split(X, y):
                train_dataset = torch.utils.data.Subset(self.data_loaders[level].dataset, train_index)
                val_dataset = torch.utils.data.Subset(self.data_loaders[level].dataset, val_index)
                val_dataset.dataset.is_training = False
                def set_worker_sharing_strategy(worker_id: int):
                    torch.multiprocessing.set_sharing_strategy("file_system")
                # Create Dataloader for Training and Validation Dataset
                kwargs = {'num_workers': self.args.num_workers_dataloader, 'pin_memory': self.args.pin_memory} if self.args.gpu else {}
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True,worker_init_fn=set_worker_sharing_strategy,**kwargs)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False,worker_init_fn=set_worker_sharing_strategy,**kwargs)
            for epoch in range(self.args.epochs):
                self.model = activate_learning_level(self.model,level=level)
                avg_train_loss = self.train(epoch_index=epoch,data_loader=train_loader,level=level)
                avg_val_loss = self.validate(epoch_index=epoch,data_loader=val_loader,level=level)
                self.tb_writer.flush()
                print(f'Epoch {epoch+1}, Level {level+1}: Average Train Loss {avg_train_loss}, Average Validation Loss {avg_val_loss}')
                # End Learning if Epoch is reached.
                if epoch == self.args.epochs-1:
                    if avg_val_loss < best_vloss:
                        best_epoch = epoch+1
                        self.best_model = copy.deepcopy(self.model)
                        best_vloss = avg_val_loss
                    print(f"Max Epoch count is reached. Best model was reached in {best_epoch}.")
                    break
                # Decay Learningrate if Step Count is reached
                if epoch % self.args.decay_steps == self.args.decay_steps-1:
                    self.scheduler.step()
                # Track best performance, and save the model's state
                if avg_val_loss < best_vloss:
                    best_epoch = epoch+1
                    self.best_model = copy.deepcopy(self.model)
                    best_vloss = avg_val_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= self.args.early_stopping_patience:
                        print(f'Early stopping triggered. Best Epoch: {best_epoch} Level: {level+1}.')
                        best_vloss = 1_000_000.
                        counter = 0
                        break
        val_loader = None
        X = [image_tuple[0] for image_tuple in self.data_loaders[0].dataset.image_label_tuple_list] 
        y = np.stack([image_tuple[1].numpy() for image_tuple in self.data_loaders[0].dataset.image_label_tuple_list])
        for train_index, val_index in msss.split(X, y):
            val_dataset = torch.utils.data.Subset(self.data_loaders[0].dataset, val_index)
            val_dataset.dataset.is_training = False
            def set_worker_sharing_strategy(worker_id: int):
                torch.multiprocessing.set_sharing_strategy("file_system")
            # Create Dataloader for Training and Validation Dataset
            kwargs = {'num_workers': self.args.num_workers_dataloader, 'pin_memory': self.args.pin_memory} if self.args.gpu else {}
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False,worker_init_fn=set_worker_sharing_strategy,**kwargs)
        self.test(epoch_index=best_epoch,data_loader=val_loader)
        model_path = os.path.join(self.path_to_model,'models',f'hmc_lmlp_{best_epoch}')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.best_model.state_dict(), model_path)
                    
    def train(self,epoch_index,data_loader,level):
        current_loss = 0.
        current_l2_loss = 0.
        current_local_loss = 0.
        last_loss = 0.
        self.model.train(True)
        num_of_train_batches = len(data_loader)
        begin, end= get_local_class_range(self.num_classes_list,level)
        for i, data in enumerate(data_loader):
            # Every data instance is an input + label pair
            inputs, labels = copy.deepcopy(data)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            labels = labels[:,begin:end]
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            inputs = inputs,level
            # Make predictions for this batch
            output,_ = self.model(inputs)

            # Compute the loss and its gradients
            x =output,labels,self.model
            loss, local_loss, l2_loss = self.criterion(x)
            
            # Clip gradients by global norm
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.norm_ratio)
            loss.backward()
            self.optimizer.step()
            
            # Gather data and report
            current_loss += loss.item()
            current_local_loss += local_loss.item()
            current_l2_loss += l2_loss.item()
            last_loss = current_loss/(i+1)
            last_local_loss = current_local_loss/(i+1)
            last_l2_loss = current_l2_loss/(i+1)
            progress_info = f"Training Level {level+1}: Epoch [{epoch_index+1}], Batch [{i+1}/{num_of_train_batches}], AVGLoss: {last_local_loss}, L2-Loss: {last_l2_loss}"
            print(progress_info, end='\r')
            tb_x = epoch_index * num_of_train_batches + i + 1
            self.tb_writer.add_scalar(f'Training/Level{level+1}/Loss', last_loss, tb_x)
            self.tb_writer.add_scalar(f'Training/Level{level+1}/LocalLoss', last_local_loss, tb_x)
            self.tb_writer.add_scalar(f'Training/Level{level+1}/L2Loss', last_l2_loss, tb_x)
        print('\n')
        return last_local_loss
    
    def validate(self,epoch_index,data_loader,level):
        current_vloss = 0.
        current_vl2_loss = 0.
        current_vlocal_loss = 0.
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        self.model.eval()
        eval_counter, eval_loss = 0, 0.0
        num_of_val_batches = len(data_loader)
        scores_list = []
        labels_list = []
        begin, end= get_local_class_range(self.num_classes_list,level)
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(data_loader):
                # Every data instance is an input + label pair
                inputs, labels = copy.deepcopy(vdata)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                labels = labels[:,begin:end]

                # Make predictions for this batch
                inputs = inputs, level
                voutput,_ = self.model(inputs)
                
                x =voutput,labels,self.model
                vloss,vlocal_loss,vl2_loss = self.criterion(x)
                
                
                scores_list.extend(voutput)
                labels_list.extend(labels)
                # Gather data and report
                current_vloss += vloss.item()
                current_vlocal_loss += vlocal_loss.item()
                current_vl2_loss += vl2_loss.item()
                last_vloss = current_vloss/(i+1)
                last_vlocal_loss = current_vlocal_loss/(i+1)
                last_vl2_loss = current_vl2_loss/(i+1)
                progress_info = f"Validation Level {level+1}: Epoch [{epoch_index+1}], Batch [{i+1}/{num_of_val_batches}], AVGLoss: {last_vlocal_loss}, L2-Loss: {last_vl2_loss}"
                print(progress_info, end='\r')
                
                tb_x = epoch_index * num_of_val_batches + eval_counter + 1
                self.tb_writer.add_scalar(f'Validation/Level{level+1}/Loss',last_vloss,tb_x)
                self.tb_writer.add_scalar(f'Validation/Level{level+1}/GlobalLoss',last_vlocal_loss,tb_x)
                self.tb_writer.add_scalar(f'Validation/Level{level+1}/L2Loss',last_vl2_loss,tb_x)            
        return last_vlocal_loss
    
    def test(self,epoch_index,data_loader):
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        self.best_model.eval()
        scores_list = []
        labels_list = []
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(data_loader):
                # Every data instance is an input + label pair
                inputs, labels = copy.deepcopy(vdata)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Make predictions for this batch
                inputs = inputs, len(self.num_classes_list)-1
                _, output_scores_list = self.best_model(inputs)
                

                # Stack the batch tensors along a new dimension (dimension 0)
                
                scores = torch.cat([batch for batch in output_scores_list],dim=1)
                scores_list.extend([score.to(dtype=torch.float64) for score in scores])
                labels_list.extend(labels)
            
        metrics_dict = dh.calc_metrics(scores_list=scores_list,labels_list=labels_list,topK=self.args.topK,pcp_hierarchy=self.pcp_hierarchy,pcp_threshold=self.args.pcp_threshold,num_classes_list=self.num_classes_list,device=self.device)
        # Save Metrics in Summarywriter.
        for key,value in metrics_dict.items():
            self.tb_writer.add_scalar(key,value,epoch_index)
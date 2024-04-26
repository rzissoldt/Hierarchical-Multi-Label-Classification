import copy,time
import torch
import sys, os
import numpy as np
sys.path.append('../')
from utils import data_helpers as dh
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit,MultilabelStratifiedKFold
import torch.optim as optim
from torchmetrics import AUROC, AveragePrecision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from HARNN.model.chmcnn_model import get_constr_out
from sklearn.metrics import average_precision_score
class BaselineTrainer():
    def __init__(self,model,criterion,optimizer,scheduler,training_dataset,num_classes_list,explicit_hierarchy,path_to_model,args,device=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.best_model = copy.deepcopy(model)
        self.explicit_hierarchy = explicit_hierarchy
        self.scheduler = scheduler
        self.args = args
        self.path_to_model = path_to_model
        self.total_class_num = sum(num_classes_list)
        self.num_classes_list = num_classes_list   
        self.tb_writer = SummaryWriter(path_to_model)
        sharing_strategy = "file_system"
        def set_worker_sharing_strategy(worker_id: int):
            torch.multiprocessing.set_sharing_strategy(sharing_strategy)
        # Create Dataloader for Training and Validation Dataset
        kwargs = {'num_workers': args.num_workers_dataloader, 'pin_memory': args.pin_memory} if self.args.gpu else {}
        self.data_loader = DataLoader(training_dataset,batch_size=args.batch_size,shuffle=True,worker_init_fn=set_worker_sharing_strategy,**kwargs)  
        
    
    def train_and_validate(self):
        counter = 0
        best_epoch = 0
        best_vloss = 1_000_000
        is_fine_tuning = False
        
        # Generate one MultiLabelStratifiedShuffleSplit for normal Training.
        train_loader,val_loader = None,None
        
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        X = [image_tuple[0] for image_tuple in self.data_loader.dataset.image_label_tuple_list] 
        y = np.stack([image_tuple[1].numpy() for image_tuple in self.data_loader.dataset.image_label_tuple_list])
        for train_index, val_index in msss.split(X, y):
            train_dataset = torch.utils.data.Subset(self.data_loader.dataset, train_index)
            val_dataset = torch.utils.data.Subset(copy.deepcopy(self.data_loader.dataset), val_index)
            val_dataset.dataset.is_training = False
            print('Train Transform:',train_dataset.dataset.is_training)
            print('Val Transform:',val_dataset.dataset.is_training)
            def set_worker_sharing_strategy(worker_id: int):
                torch.multiprocessing.set_sharing_strategy("file_system")
            # Create Dataloader for Training and Validation Dataset
            kwargs = {'num_workers': self.args.num_workers_dataloader, 'pin_memory': self.args.pin_memory} if self.args.gpu else {}
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True,worker_init_fn=set_worker_sharing_strategy,**kwargs)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False,worker_init_fn=set_worker_sharing_strategy,**kwargs)
        #self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=len(train_loader)/(self.args.batch_size*self.args.epochs))
        #T_0 = 10
        #T_mult = 2
        #self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0, T_mult)
        
        for epoch in range(self.args.epochs):
            avg_train_loss = self.train(epoch_index=epoch,data_loader=train_loader)
            avg_val_loss = self.validate(epoch_index=epoch,data_loader=val_loader)
            self.tb_writer.flush()
            print(f'Epoch {epoch+1}: Train Loss {avg_train_loss}, Validation Loss {avg_val_loss}')
            
            # End Training if Max Epoch is reached
            if epoch == self.args.epochs-1:
                if avg_val_loss < best_vloss:
                    best_epoch = epoch+1
                    self.best_model = copy.deepcopy(self.model)
                    best_vloss = avg_val_loss
                print(f"Max Epoch count is reached. Best model was reached in {best_epoch}.")
                break     
            # Track best performance, and save the model's state
            if avg_val_loss < best_vloss:
                best_epoch = epoch+1
                self.best_model = copy.deepcopy(self.model)
                best_vloss = avg_val_loss
                counter = 0
            else:
                counter += 1
                if counter >= self.args.early_stopping_patience and not is_fine_tuning:
                    print(f'Early stopping triggered and validate best Epoch {best_epoch}.')
                    print(f'Begin fine tuning model.')
                    self.model = copy.deepcopy(self.best_model)
                    self.unfreeze_backbone()
                    #self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=len(train_loader)/(self.args.batch_size*self.args.epochs))
                    T_0 = self.scheduler.T_0
                    T_mult = self.scheduler.T_mult
                    self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0, T_mult)
                    is_fine_tuning = True
                    counter = 0
                    continue
                if counter >= self.args.early_stopping_patience and is_fine_tuning:
                    print(f'Early stopping triggered in fine tuning Phase. {best_epoch} was the best Epoch.')
                    break
        # Test and save Best Model
        self.model = copy.deepcopy(self.best_model)
        self.test(epoch_index=best_epoch,data_loader=val_loader)
        model_path = os.path.join(self.path_to_model,'models',f'baseline_model_{best_epoch}')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        self.tb_writer.flush()
        
    
    def train(self,epoch_index,data_loader):
        current_loss = 0.
        current_global_loss = 0.
        current_l2_loss = 0.
        last_loss = 0.
        self.model.train(True)
        predicted_list = []
        labels_list = []
        num_of_train_batches = len(data_loader)
        
        for i, data in enumerate(data_loader):
            start_time = time.perf_counter()
            # Every data instance is an input + label pair
            t1 = time.perf_counter()
            #inputs, labels = copy.deepcopy(data)
            inputs, labels = data
            t2 = time.perf_counter()
            print(f'Time for copy: {t2-t1:.4f} seconds.')
            t1 = time.perf_counter()
            inputs = inputs.to(self.device)
            torch.cuda.synchronize()
            labels = labels.to(self.device)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            print(f'Time to GPU: {t2-t1:.4f} seconds.')
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
           
            # Make predictions for this batch
            output = self.model(inputs.float())
            torch.cuda.synchronize()
            # Compute the loss and its gradients
            
            x = output,labels.double()
            loss = self.criterion(x)
            predicted = output.data > 0.5            

            t1 = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            print(f'Time for backward: {t2-t1:.4f} seconds.')            
            # Clip gradients by global norm
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.norm_ratio)
            t1 = time.perf_counter()
            self.optimizer.step()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            print(f'Time for optimizer step: {t2-t1:.4f} seconds.')
            self.scheduler.step(epoch_index+i/num_of_train_batches)
            
            t1 = time.perf_counter()
            
            #print(learning_rates)
            predicted_list.extend(predicted)
            labels_list.extend(labels)
            # Gather data and report
            current_loss += loss.detach()
            last_loss = current_loss/(i+1)
            learning_rates = [str(param_group['lr']) for param_group in self.optimizer.param_groups]
            #learning_rates_str = 'LR: ' + ', '.join(learning_rates)
            progress_info = f"Training: Epoch [{epoch_index+1}], Batch [{i+1}/{num_of_train_batches}]"#, AVGLoss: {last_loss}", {learning_rates_str}"
            
            print(progress_info, end='\n')
            tb_x = epoch_index * num_of_train_batches + i + 1
            
            #for i in range(len(learning_rates)):
            #    self.tb_writer.add_scalar(f'Training/LR{i}', float(learning_rates[i]), tb_x)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            print(f'Time for metrics step: {t2-t1:.4f} seconds.')
           
            print(progress_info, end='\r')
            end_time = time.perf_counter()
            print(f'Time for batch step: {end_time-start_time:.4f} seconds.')
        # Gather data and report
        self.tb_writer.add_scalar('Training/Loss', last_loss, tb_x)
        auprc = AveragePrecision(task="binary")
        predicted_onehot_labels = torch.cat([torch.unsqueeze(tensor,0) for tensor in predicted_list],dim=0).to(self.device)
        labels = torch.cat([torch.unsqueeze(tensor,0) for tensor in labels_list],dim=0).to(self.device)
        progress_info = f"Training: Epoch [{epoch_index+1}], Loss: {last_loss.item()}"
        
        
        
        print('\n')
        return last_loss
    
    def validate(self,epoch_index,data_loader):
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        self.model.eval()
        current_vloss = 0.
        current_vglobal_loss = 0.
        current_vl2_loss = 0.
        last_vloss = 0.
        eval_counter = 0
        num_of_val_batches = len(data_loader)
        predicted_list = []
        labels_list = []
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(data_loader):
                # Every data instance is an input + label pair
                inputs, labels = copy.deepcopy(vdata)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)


                # Make predictions for this batch
                output = self.model(inputs.float())

                # Compute the loss and its gradients
                x = output,labels.double()
                loss = self.criterion(x)
                predicted = output.data > 0.5
                predicted_list.extend(predicted)
                labels_list.extend(labels)
                # Gather data and report
                current_vloss += loss.item()
                last_vloss = current_vloss/(i+1)
                progress_info = f"Validation: Epoch [{epoch_index+1}], Batch [{i+1}/{num_of_val_batches}], AVGLoss: {last_vloss}"
                print(progress_info, end='\r')
                tb_x = epoch_index * num_of_val_batches + i + 1
                self.tb_writer.add_scalar('Validation/Loss', last_vloss, tb_x)
            
        return last_vloss
    
    def test(self,epoch_index,data_loader):
        print(f"Evaluating best model of epoch {epoch_index}.")
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
                constr_output = self.best_model(inputs.float())
                scores_list.extend(constr_output)
                labels_list.extend(labels) 
        metrics_dict = dh.calc_metrics(scores_list=scores_list,labels_list=labels_list,threshold=self.args.threshold,topK=self.args.topK,pcp_hierarchy=self.explicit_hierarchy.to('cpu').numpy(),pcp_threshold=self.args.pcp_threshold,num_classes_list=self.num_classes_list,device=self.device,eval_pcp=self.args.pcp_metrics_active,eval_hierarchical_metrics=True)
        # Save Metrics in Summarywriter.
        for key,value in metrics_dict.items():
            self.tb_writer.add_scalar(key,value,epoch_index)
            
    def unfreeze_backbone(self):
        """
        Unfreezes the backbone of the model and splits the learning rate into three different parts.


        Returns:
        - None
        """
        # Set the requires_grad attribute of the backbone parameters to True
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        
        optimizer_dict = self.optimizer.param_groups[0]
        
        param_groups = [copy.deepcopy(optimizer_dict) for i in range(3)]
        # Get the parameters of the model
        backbone_model_params = list(self.model.backbone.parameters())

        # Calculate the number of parameters for each section
        first_backbone_params = int(0.8 * len(backbone_model_params))

        # Assign learning rates to each parameter group
        base_lr = self.args.learning_rate*1e-1
        current_lr = param_groups[0]['lr']
        param_groups[0]['params'] = backbone_model_params[:first_backbone_params]
        param_groups[0]['lr'] = base_lr * 1e-4
        param_groups[0]['initial_lr'] = base_lr * 1e-4
        param_groups[1]['params'] = backbone_model_params[first_backbone_params:]
        param_groups[1]['lr'] = base_lr * 1e-2
        param_groups[1]['initial_lr'] = base_lr * 1e-2
        param_groups[2]['params'] = list(self.model.fc.parameters())
        param_groups[2]['lr'] = base_lr
        param_groups[2]['initial_lr'] = base_lr
        
        # Update the optimizer with the new parameter groups
        self.optimizer.param_groups = param_groups
        
        
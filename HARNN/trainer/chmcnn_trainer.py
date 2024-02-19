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
from HARNN.model.chmcnn_model import get_constr_out
from sklearn.metrics import average_precision_score
class CHMCNNTrainer():
    def __init__(self,model,criterion,optimizer,scheduler,training_dataset,num_classes_list,explicit_hierarchy,pcp_hierarchy,path_to_model,args,device=None):
        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = device
        self.best_model = copy.deepcopy(model)
        self.explicit_hierarchy = explicit_hierarchy
        self.pcp_hierarchy = pcp_hierarchy
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
        best_vauprc = 0.
        is_fine_tuning = False
        
        # Generate one MultiLabelStratifiedShuffleSplit for normal Training.
        train_loader,val_loader = None,None
        
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        X = [image_tuple[0] for image_tuple in self.data_loader.dataset.image_label_tuple_list] 
        y = np.stack([image_tuple[1].numpy() for image_tuple in self.data_loader.dataset.image_label_tuple_list])
        for train_index, val_index in msss.split(X, y):
            train_dataset = torch.utils.data.Subset(self.data_loader.dataset, train_index)
            val_dataset = torch.utils.data.Subset(self.data_loader.dataset, val_index)
            val_dataset.dataset.is_training = False
            def set_worker_sharing_strategy(worker_id: int):
                torch.multiprocessing.set_sharing_strategy("file_system")
            # Create Dataloader for Training and Validation Dataset
            kwargs = {'num_workers': self.args.num_workers_dataloader, 'pin_memory': self.args.pin_memory} if self.args.gpu else {}
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True,worker_init_fn=set_worker_sharing_strategy,**kwargs)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False,worker_init_fn=set_worker_sharing_strategy,**kwargs)
        for epoch in range(self.args.epochs):
            avg_train_auprc = self.train(epoch_index=epoch,data_loader=train_loader)
            avg_val_aurpc = self.validate(epoch_index=epoch,data_loader=val_loader)
            self.tb_writer.flush()
            print(f'Epoch {epoch+1}: Train AUPRC {avg_train_auprc}, Validation AUPRC {avg_val_aurpc}')
            
            # End Training if Max Epoch is reached
            if epoch == self.args.epochs-1:
                if avg_val_aurpc > best_vauprc:
                    best_epoch = epoch+1
                    self.best_model = copy.deepcopy(self.model)
                    best_vauprc = avg_val_aurpc
                print(f"Max Epoch count is reached. Best model was reached in {best_epoch}.")
                break
            # Decay Learningrate if Step Count is reached
            if epoch % self.args.decay_steps == self.args.decay_steps-1:
                self.scheduler.step()
            # Track best performance, and save the model's state
            if avg_val_aurpc > best_vauprc:
                best_epoch = epoch+1
                self.best_model = copy.deepcopy(self.model)
                best_vauprc = avg_val_aurpc
                counter = 0
            else:
                counter += 1
                if counter >= self.args.early_stopping_patience and not is_fine_tuning:
                    print(f'Early stopping triggered and validate best Epoch {best_epoch}.')
                    print(f'Begin fine tuning model.')
                    self.unfreeze_backbone()
                    best_vauprc = 0.
                    is_fine_tuning = True
                    counter = 0
                    continue
                if counter >= self.args.early_stopping_patience and is_fine_tuning:
                    print(f'Early stopping triggered in fine tuning Phase. {best_epoch} was the best Epoch.')
                    break
        # Test and save Best Model
        self.model = copy.deepcopy(self.best_model)
        self.test(epoch_index=best_epoch,data_loader=val_loader)
        model_path = os.path.join(self.path_to_model,'models',f'chmcnn_{best_epoch}')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        self.tb_writer.flush()
        
    def train_and_validate_k_crossfold(self,k_folds=5):
        counter = 0
        best_epoch = 0
        best_vloss = 1_000_000.
        epoch = 0
        is_fine_tuning = False
        is_finished = False
        mskf = MultilabelStratifiedKFold(n_splits=k_folds,shuffle=True,random_state=42)
        X = [image_tuple[0] for image_tuple in self.data_loader.dataset.image_label_tuple_list]
        y = np.stack([image_tuple[1].numpy() for image_tuple in self.data_loader.dataset.image_label_tuple_list])
        while epoch < self.args.epochs and not is_finished:
            for fold, (train_index, val_index) in enumerate(mskf.split(X, y)):
                train_dataset = torch.utils.data.Subset(self.data_loader.dataset, train_index)
                val_dataset = torch.utils.data.Subset(self.data_loader.dataset, val_index)
                val_dataset.dataset.is_training = False
                def set_worker_sharing_strategy(worker_id: int):
                    torch.multiprocessing.set_sharing_strategy("file_system")
                # Create Dataloader for Training and Validation Dataset
                kwargs = {'num_workers': self.args.num_workers_dataloader, 'pin_memory': self.args.pin_memory} if self.args.gpu else {}
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True,worker_init_fn=set_worker_sharing_strategy,**kwargs)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False,worker_init_fn=set_worker_sharing_strategy,**kwargs)

                print(f"Epoch {epoch + 1}/{self.args.epochs}, Fold {fold + 1}/{k_folds}:")
                avg_train_loss = self.train(epoch_index=epoch, data_loader=train_loader)
                calc_metrics = epoch == self.args.epochs - 1
                avg_val_loss = self.validate(epoch_index=epoch, data_loader=val_loader)
                self.tb_writer.flush()
                print(f'Epoch {epoch+1}: Average Train Loss {avg_train_loss}, Average Validation Loss {avg_val_loss}')
                epoch+=1
                
                
                # Decay Learning rate if Step Count is reached
                if epoch % self.args.decay_steps == self.args.decay_steps - 1:
                    self.scheduler.step()

                # Track best performance, and save the model's state
                if avg_val_loss < best_vloss:
                    best_epoch = epoch+1
                    self.best_model = copy.deepcopy(self.model)
                    best_vloss = avg_val_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= self.args.early_stopping_patience and not is_fine_tuning:
                        print(f'Early stopping triggered and validate best Epoch {best_epoch + 1}.')
                        print(f'Begin fine-tuning model.')
                        self.unfreeze_backbone()
                        best_vloss = 1_000_000.
                        is_fine_tuning = True
                        counter = 0
                        continue
                    if counter >= self.args.early_stopping_patience and is_fine_tuning:
                        print(f'Early stopping triggered in fine-tuning Phase. {best_epoch + 1} was the best Epoch.')
                        print(f'Validate fine-tuned Model.')
                        is_finished = True
                        break
        # Test and save Best Model
        self.test(epoch_index=best_epoch,data_loader=val_loader)
        model_path = os.path.join(self.path_to_model,'models',f'hmcnet_{best_epoch}')
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
            # Every data instance is an input + label pair
            inputs, labels = copy.deepcopy(data)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            output = self.model(inputs.float())

            # Compute the loss and its gradients
            constr_output = get_constr_out(output, self.explicit_hierarchy)
            train_output = labels*output.double()
            train_output = get_constr_out(train_output, self.explicit_hierarchy)
            train_output = (1-labels)*constr_output.double() + labels*train_output
            x = train_output,labels.double(),self.model
            loss,global_loss,l2_loss = self.criterion(x)
            predicted = constr_output.data > 0.5
            
            # Total number of labels
            total_train = labels.size(0) * labels.size(1)
            # Total correct predictions
            correct_train = (predicted == labels.byte()).sum()

            # Clip gradients by global norm
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.norm_ratio)
            loss.backward()
            self.optimizer.step()
            predicted_list.extend(predicted)
            labels_list.extend(labels)
            # Gather data and report
            current_loss += loss.item()
            current_global_loss += global_loss.item()
            current_l2_loss += l2_loss.item()
            last_loss = current_loss/(i+1)
            last_global_loss = current_global_loss/(i+1)
            last_l2_loss = current_l2_loss/(i+1)
            progress_info = f"Training: Epoch [{epoch_index+1}], Batch [{i+1}/{num_of_train_batches}], AVGLoss: {last_global_loss}, L2-Loss: {last_l2_loss}"
            print(progress_info, end='\r')
            tb_x = epoch_index * num_of_train_batches + i + 1
            self.tb_writer.add_scalar('Training/Loss', last_loss, tb_x)
            self.tb_writer.add_scalar('Training/GlobalLoss', last_global_loss, tb_x)
            self.tb_writer.add_scalar('Training/L2Loss', last_l2_loss, tb_x)
        # Gather data and report
        auprc = AveragePrecision(task="binary")
        predicted_onehot_labels = torch.cat([torch.unsqueeze(tensor,0) for tensor in predicted_list],dim=0).to(self.device)
        labels = torch.cat([torch.unsqueeze(tensor,0) for tensor in labels_list],dim=0).to(self.device)
        eval_auprc = auprc(predicted_onehot_labels.to(dtype=torch.float32),labels.to(dtype=torch.long))
        progress_info = f"Training: Epoch [{epoch_index+1}], AUPRC: {eval_auprc}"
        
        
        self.tb_writer.add_scalar('Training/AUPRC',eval_auprc,epoch_index)
        print('\n')
        return eval_auprc
    
    def validate(self,epoch_index,data_loader):
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        self.model.eval()
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
                constr_output = self.model(inputs.float())
                predicted = constr_output > 0.5
                predicted_list.extend(predicted)
                labels_list.extend(labels)
            # Gather data and report
            
            auprc = AveragePrecision(task="binary")
            predicted_onehot_labels = torch.cat([torch.unsqueeze(tensor,0) for tensor in predicted_list],dim=0).to(self.device)
            labels = torch.cat([torch.unsqueeze(tensor,0) for tensor in labels_list],dim=0).to(self.device)
            eval_auprc = auprc(predicted_onehot_labels.to(dtype=torch.float32),labels.to(dtype=torch.long))
            progress_info = f"Validation: Epoch [{epoch_index+1}], AUPRC: {eval_auprc}"
            print(progress_info, end='\r')
            tb_x = epoch_index * num_of_val_batches + eval_counter + 1
            self.tb_writer.add_scalar('Validation/AUPRC',eval_auprc,tb_x)
                
        return eval_auprc
    
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
        metrics_dict = dh.calc_metrics(scores_list=scores_list,labels_list=labels_list,topK=self.args.topK,pcp_hierarchy=self.pcp_hierarchy,pcp_threshold=self.args.pcp_threshold,num_classes_list=self.num_classes_list,device=self.device)
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
        
        param_groups = [copy.deepcopy(optimizer_dict) for i in range(4)]
        # Get the parameters of the model
        backbone_model_params = list(self.model.backbone.parameters())

        # Calculate the number of parameters for each section
        first_backbone_params = int(0.2 * len(backbone_model_params))

        # Assign learning rates to each parameter group
        base_lr = optimizer_dict['lr']
        param_groups[0]['params'] = backbone_model_params[:first_backbone_params]
        param_groups[0]['lr'] = base_lr * 1e-4
        param_groups[1]['params'] = backbone_model_params[first_backbone_params:]
        param_groups[1]['lr'] = base_lr * 1e-2
        param_groups[2]['params'] = self.model.fc.parameters()
        param_groups[2]['lr'] = base_lr
        
        # Update the optimizer with the new parameter groups
        self.optimizer.param_groups = param_groups
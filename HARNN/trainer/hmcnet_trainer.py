import copy, datetime
import torch
import sys, os, time
import numpy as np
sys.path.append('../')
from utils import data_helpers as dh
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit,MultilabelStratifiedKFold
import torch.optim as optim
from torchmetrics import AUROC, AveragePrecision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
class HmcNetTrainer():
    def __init__(self,model,criterion,optimizer,scheduler,training_dataset,explicit_hierarchy,num_classes_list,path_to_model,args,device=None):
        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = device
        self.best_model = copy.deepcopy(model)
        self.explicit_hierarchy= explicit_hierarchy
        self.args = args
        self.num_classes_list = num_classes_list
        self.path_to_model = path_to_model        
        self.tb_writer = SummaryWriter(path_to_model)
        sharing_strategy = "file_system"
        def set_worker_sharing_strategy(worker_id: int):
            torch.multiprocessing.set_sharing_strategy(sharing_strategy)
        # Create Dataloader for Training and Validation Dataset
        kwargs = {'num_workers': args.num_workers_dataloader, 'pin_memory': args.pin_memory} if self.args.gpu else {}
        self.data_loader = DataLoader(training_dataset,batch_size=args.batch_size,shuffle=True,worker_init_fn=set_worker_sharing_strategy,**kwargs)  
        
    
    def train_and_validate(self):
        start_time = time.time()
        counter = 0
        best_epoch = 0
        best_vloss = 1_000_000.
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
        for epoch in range(self.args.epochs):
            avg_train_loss = self.train(epoch_index=epoch,data_loader=train_loader)
            avg_val_loss = self.validate(epoch_index=epoch,data_loader=val_loader)
            self.tb_writer.flush()
            print(f'Epoch {epoch+1}: Average Train Loss {avg_train_loss}, Average Validation Loss {avg_val_loss}')
            
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
                    T_0 = self.scheduler.T_0
                    T_mult = self.scheduler.T_mult
                    self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0, T_mult)
                    is_fine_tuning = True
                    counter = 0
                    continue
                if counter >= self.args.early_stopping_patience and is_fine_tuning:
                    print(f'Early stopping triggered in fine tuning Phase. {best_epoch} was the best Epoch.')
                    break
        end_time = time.time()
        training_time = end_time - start_time
        self.tb_writer.add_scalar('Training/Time',training_time)        
        # Test and save Best Model
        self.test(epoch_index=best_epoch,data_loader=val_loader)
        model_path = os.path.join(self.path_to_model,'models',f'hmcnet_{best_epoch}')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        self.tb_writer.flush()
    
    
                      
    def train(self,epoch_index,data_loader):
        current_loss = 0.
        current_global_loss = 0.
        current_local_loss = 0.
        current_hierarchy_loss = 0.
        current_l2_loss = 0.
        last_loss = 0.
        self.model.train(True)
        num_of_train_batches = len(data_loader)
        for i, data in enumerate(data_loader):
            # Every data instance is an input + label pair
            inputs, labels = copy.deepcopy(data)
            inputs = inputs.to(self.device)
            y_total_onehot = labels[0].to(self.device)
            y_local_onehots = [label.to(self.device) for label in labels[1:]]
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            _, local_scores_list, global_logits = self.model(inputs)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            print(f'Inference: {t2-t1:.5f}s')
            # Compute the loss and its gradients
            predictions = (local_scores_list,global_logits)
            targets = (y_local_onehots,y_total_onehot)
            x = (predictions,targets)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            loss,global_loss,local_loss,hierarchy_loss = self.criterion(x)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            print(f'Loss:{t2-t1:.5f}s')
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            print(f'Backward:{t2-t1:.5f}s')
            # Adjust learning weights
            self.optimizer.step()
            self.scheduler.step(epoch_index+i/num_of_train_batches)
            learning_rates = [str(param_group['lr']) for param_group in self.optimizer.param_groups]
            learning_rates_str = 'LR: ' + ', '.join(learning_rates)
            # Gather data and report
            current_loss += loss.detach()
            current_global_loss += global_loss.detach()
            current_local_loss += local_loss.detach()
            current_hierarchy_loss += hierarchy_loss.detach()
            last_loss = current_loss/(i+1)
            last_global_loss = current_global_loss/(i+1)
            last_local_loss = current_local_loss/(i+1)
            last_hierarchy_loss = current_hierarchy_loss/(i+1)
            progress_info = f"Training: Epoch [{epoch_index+1}], Batch [{i+1}/{num_of_train_batches}] {learning_rates_str}"
            print(progress_info, end='\r')
            tb_x = epoch_index * num_of_train_batches + i + 1
            self.tb_writer.add_scalar('Training/Loss', last_loss, tb_x)
            self.tb_writer.add_scalar('Training/GlobalLoss', last_global_loss, tb_x)
            self.tb_writer.add_scalar('Training/LocalLoss', last_local_loss, tb_x)
            self.tb_writer.add_scalar('Training/HierarchyLoss', last_hierarchy_loss, tb_x)
            for i in range(len(learning_rates)):
                self.tb_writer.add_scalar(f'Training/LR{i}', float(learning_rates[i]), tb_x)
        print('\n')
        return last_global_loss+last_local_loss+last_hierarchy_loss
    
    def validate(self,epoch_index,data_loader):
        running_vloss = 0.0
        current_vglobal_loss = 0.
        current_vlocal_loss = 0.
        current_vhierarchy_loss = 0.
        current_vl2_loss = 0.
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        self.model.eval()
        eval_counter, eval_loss = 0, 0.0
        num_of_val_batches = len(data_loader)
        scores_list = []
        true_onehot_labels_list = []
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(data_loader):
                vinputs, vlabels = copy.deepcopy(vdata)
                vinputs = vinputs.to(self.device)
                y_total_onehot = vlabels[0].to(self.device)
                y_local_onehots = [label.to(self.device) for label in vlabels[1:]]
                # Make predictions for this batch
                scores, local_scores_list, global_logits = self.model(vinputs)

                # Compute the loss and its gradients
                predictions, targets = (local_scores_list,global_logits),(y_local_onehots,y_total_onehot)
                x = (predictions,targets)
                vloss,vglobal_loss,vlocal_loss,vhierarchy_loss = self.criterion(x)
                current_vglobal_loss += vglobal_loss.item()
                current_vlocal_loss += vlocal_loss.item()
                current_vhierarchy_loss += vhierarchy_loss.item()
                
                last_vglobal_loss = current_vglobal_loss/(i+1)
                last_vlocal_loss = current_vlocal_loss/(i+1)
                last_vhierarchy_loss = current_vhierarchy_loss/(i+1)
                
                running_vloss += vloss.item()
                for j in scores:
                    scores_list.append(j)
                # Convert each tensor to a list of lists
                for i in y_total_onehot:
                    true_onehot_labels_list.append(i)                
                eval_loss = running_vloss/(eval_counter+1)
                #progress_info = f'Validation: Epoch [{epoch_index+1}], Batch [{eval_counter+1}/{num_of_val_batches}], AVGLoss: {eval_loss}'
                progress_info = f"Validation: Epoch [{epoch_index+1}], Batch [{eval_counter+1}/{num_of_val_batches}], AVGLoss: {last_vglobal_loss+last_vlocal_loss+last_vhierarchy_loss}"
                print(progress_info, end='\r')
                
                tb_x = epoch_index * num_of_val_batches + eval_counter + 1
                self.tb_writer.add_scalar('Validation/Loss',eval_loss,tb_x)
                self.tb_writer.add_scalar('Validation/GlobalLoss',last_vglobal_loss,tb_x)
                self.tb_writer.add_scalar('Validation/LocalLoss',last_vlocal_loss,tb_x)
                self.tb_writer.add_scalar('Validation/HierarchyLoss',last_vhierarchy_loss,tb_x)
                eval_counter+=1
            print('\n')                
            return last_vglobal_loss+last_vlocal_loss+last_vhierarchy_loss
    
    def test(self,epoch_index,data_loader):
        print(f"Evaluating best model of epoch {epoch_index}.")
        scores_list = []
        true_onehot_labels_list = []
        self.best_model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(data_loader):
                vinputs, vlabels = copy.deepcopy(vdata)
                vinputs = vinputs.to(self.device)
                y_total_onehot = vlabels[0].to(self.device)
                y_local_onehots = [label.to(self.device) for label in vlabels[1:]]
                # Make predictions for this batch
                scores, local_scores_list, global_logits = self.best_model(vinputs)
                for j in scores:
                    scores_list.append(j)
                # Convert each tensor to a list of lists
                for i in y_total_onehot:
                    true_onehot_labels_list.append(i)                
        metrics_dict = dh.calc_metrics(scores_list=scores_list,threshold=self.args.threshold,labels_list=true_onehot_labels_list,topK=self.args.topK,pcp_hierarchy=self.explicit_hierarchy,pcp_threshold=self.args.pcp_threshold,num_classes_list=self.num_classes_list,device=self.device,eval_pcp=self.args.pcp_metrics_active,eval_hierarchical_metrics=True)
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
        param_groups[2]['params'] = list(self.model.ham_modules.parameters())
        param_groups[2]['lr'] = base_lr
        param_groups[2]['initial_lr'] = base_lr
        param_groups[3]['params'] = list(self.model.hybrid_predicting_module.parameters())
        param_groups[3]['lr'] = base_lr
        param_groups[3]['initial_lr'] = base_lr
        
        # Update the optimizer with the new parameter groups
        self.optimizer.param_groups = param_groups
        
    """def unfreeze_backbone(self):
        
        Unfreezes the backbone of the model and splits the learning rate into three different parts.


        Returns:
        - None
        
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
        param_groups[2]['params'] = list(self.model.ham_modules.parameters())
        param_groups[2]['lr'] = base_lr
        param_groups[3]['params'] = list(self.model.hybrid_predicting_module.parameters())
        param_groups[3]['lr'] = base_lr
        
        

        # Update the optimizer with the new parameter groups
        self.optimizer.param_groups = param_groups"""
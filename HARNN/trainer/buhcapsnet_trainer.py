import copy, datetime
import torch
import sys, os
import numpy as np
import torch.optim as optim
sys.path.append('../')
from utils import data_helpers as dh
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit,MultilabelStratifiedKFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MultilabelAveragePrecision
class BUHCapsNetTrainer():
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
        #lambda_values = self.lambda_updater.get_lambda_values()
        #self.criterion.initial_loss_weights = lambda_values
    def train_and_validate(self):
        counter = 0
        best_epoch = 0
        best_vloss = 1_000_000
        #best_vloss = 1_000_000
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
            print(f'Epoch {epoch+1}: Train Loss {avg_train_loss}, Validation Loss {avg_val_loss}')
            
            # End Training if Max Epoch is reached
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
                if counter >= self.args.early_stopping_patience and not is_fine_tuning:
                    print(f'Early stopping triggered and validate best Epoch {best_epoch}.')
                    print(f'Begin fine tuning model.')
                    self.model = copy.deepcopy(self.best_model)
                    self.unfreeze_backbone()
                    is_fine_tuning = True
                    counter = 0
                    T_0 = self.scheduler.T_0
                    T_mult = self.scheduler.T_mult
                    self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0, T_mult)
                    continue
                if counter >= self.args.early_stopping_patience and is_fine_tuning:
                    print(f'Early stopping triggered in fine tuning Phase. {best_epoch} was the best Epoch.')
                    break
        # Test and save Best Model
        self.model = copy.deepcopy(self.best_model)
        self.test(epoch_index=best_epoch,data_loader=val_loader)
        model_path = os.path.join(self.path_to_model,'models',f'buhcapsnet_{best_epoch}')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        self.tb_writer.flush()
    
    
                      
    def train(self,epoch_index,data_loader):
        current_margin_loss = 0.
        self.model.train(True)
        
        num_of_train_batches = len(data_loader)
        for i, data in enumerate(data_loader):
            # Every data instance is an input + label pair
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            inputs, labels = data
            inputs= inputs.to(self.device)
            y_local_onehots = [label.to(self.device) for label in labels]
            end.record()
            torch.cuda.synchronize()

            print('To GPU:',start.elapsed_time(end))
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            
            # Make predictions for this batch
            start.record()
            
            local_scores = self.model(inputs)
            end.record()
            torch.cuda.synchronize()
            

            print('To Forward:',start.elapsed_time(end))
            # Compute the loss and its gradients            
            x = (local_scores,y_local_onehots)
            start.record()
            margin_loss = self.criterion(x)
            end.record()
            torch.cuda.synchronize()
            print('To Loss:',start.elapsed_time(end))
            start.record()
            margin_loss.backward()
            end.record()
            torch.cuda.synchronize()
            print('To Backwardpass:',start.elapsed_time(end))
            # Clip gradients by global norm
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.norm_ratio)
            
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            self.scheduler.step(epoch_index+i/num_of_train_batches)
            current_margin_loss += margin_loss.detach()
            
            if i % 32 == 0:
                progress_info = f"Training: Epoch [{epoch_index+1}], Batch [{i+1}/{num_of_train_batches}]"
                print(progress_info, end='\r')
        last_margin_loss = current_margin_loss/num_of_train_batches
            
            
        #tb_x = epoch_index * num_of_train_batches + i + 1
        self.tb_writer.add_scalar('Training/MarginLoss', last_margin_loss, epoch_index)
                      
        print('\n')
        return  last_margin_loss
    
    def validate(self,epoch_index,data_loader):
        current_global_loss = 0.
        current_margin_loss = 0.
        current_l2_loss = 0.
        scores_list = []
        true_onehot_labels_list = []
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        self.model.eval()
        eval_counter = 0
        num_of_val_batches = len(data_loader)
        
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(data_loader):
                vinputs, vlabels = copy.deepcopy(vdata)
                vinputs = vinputs.to(self.device)
                y_local_onehots = [label.to(self.device) for label in vlabels]
                y_global_onehots = torch.cat(y_local_onehots,dim=1)
                # Zero your gradients for every batch!
                self.optimizer.zero_grad()
                
                # Make predictions for this batch
                local_scores = self.model(vinputs)

                # Compute the loss and its gradients            
                x = (local_scores,y_local_onehots)
                vmargin_loss = self.criterion(x)
            
                current_margin_loss += vmargin_loss
                
                last_vmargin_loss = current_margin_loss/(i+1)
                
                   
                progress_info = f"Validation: Epoch [{epoch_index+1}], Batch [{i+1}/{num_of_val_batches}]"
                print(progress_info, end='\r')
                
                tb_x = epoch_index * num_of_val_batches + eval_counter + 1
                self.tb_writer.add_scalar('Validation/MarginLoss',last_vmargin_loss,tb_x)
                
                eval_counter+=1
                global_scores = torch.cat(local_scores,dim=1)
                for j in global_scores:
                    scores_list.append(j)
                # Convert each tensor to a list of lists
                for i in y_global_onehots:
                    true_onehot_labels_list.append(i)
            print('\n')
            scores = torch.cat([torch.unsqueeze(tensor,0) for tensor in scores_list],dim=0).to('cpu')
            true_onehot_labels = torch.cat([torch.unsqueeze(tensor,0) for tensor in true_onehot_labels_list],dim=0).to('cpu')
                 
            macro_aurpc_per_layer=dh.get_per_layer_auprc(scores=scores,labels=true_onehot_labels,num_classes_list=self.num_classes_list)
            macro_auprc = MultilabelAveragePrecision(num_labels=sum(self.num_classes_list),average='macro')
            eval_macro_auprc_layer = macro_auprc(scores.to(dtype=torch.float32),true_onehot_labels.to(dtype=torch.long))
            self.criterion.update_loss_weights(macro_aurpc_per_layer)
            print(f'Current Loss Weights: {self.criterion.current_loss_weights}')             
            return last_vmargin_loss
        
    def test(self,epoch_index,data_loader):
        print(f"Evaluating best model of epoch {epoch_index}.")
        scores_list = []
        true_onehot_labels_list = []
        self.best_model.eval()
        
        with torch.no_grad():
            for i, vdata in enumerate(data_loader):
                vinputs, vlabels = copy.deepcopy(vdata)
                vinputs= vinputs.to(self.device)
                y_local_onehots = [label.to(self.device) for label in vlabels]
                y_global_onehots = torch.cat(y_local_onehots,dim=1)
                # Zero your gradients for every batch!
                self.optimizer.zero_grad()
                # Make predictions for this batch
                local_scores = self.model(vinputs)
                global_scores = torch.cat(local_scores,dim=1)
                for j in global_scores:
                    scores_list.append(j)
                # Convert each tensor to a list of lists
                for i in y_global_onehots:
                    true_onehot_labels_list.append(i)
                   
        metrics_dict = dh.calc_metrics(scores_list=scores_list,labels_list=true_onehot_labels_list,topK=self.args.topK,pcp_hierarchy=self.explicit_hierarchy,pcp_threshold=self.args.pcp_threshold,num_classes_list=self.num_classes_list,device=self.device,eval_pcp=self.args.pcp_metrics_active,threshold=self.args.threshold,eval_hierarchical_metrics=True)
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
        base_lr = self.args.learning_rate
        current_lr = param_groups[0]['lr']
        param_groups[0]['params'] = backbone_model_params[:first_backbone_params]
        param_groups[0]['lr'] = base_lr * 1e-4
        param_groups[0]['initial_lr'] = base_lr * 1e-4
        param_groups[1]['params'] = backbone_model_params[first_backbone_params:]
        param_groups[1]['lr'] = base_lr * 1e-2
        param_groups[1]['initial_lr'] = base_lr * 1e-2
        param_groups[2]['params'] = list(self.model.secondary_capsules.parameters())
        param_groups[2]['lr'] = base_lr
        
        # Update the optimizer with the new parameter groups
        self.optimizer.param_groups = param_groups
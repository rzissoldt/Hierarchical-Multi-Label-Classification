import copy, datetime
import torch
import sys, os
import numpy as np
sys.path.append('../')
from utils import data_helpers as dh
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit,MultilabelStratifiedKFold

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
class HCapsNetTrainer():
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
            val_dataset = torch.utils.data.Subset(self.data_loader.dataset, val_index)
            val_dataset.dataset.is_training = False
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
                    print(f'Early stopping {best_epoch} was the best Epoch.')
                    print(f'Validate Model.')
                    #avg_val_loss = self.validate(epoch_index=epoch, data_loader=val_loader)
                    break
        # Test and save Best Model
        self.test(epoch_index=best_epoch,data_loader=val_loader)
        model_path = os.path.join(self.path_to_model,'models',f'hcapsnet_{best_epoch}')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        self.tb_writer.flush()
    
    
                      
    def train(self,epoch_index,data_loader):
        current_global_loss = 0.
        current_margin_loss = 0.
        current_reconstruction_loss = 0.
        current_l2_loss = 0.
        last_loss = 0.
        self.model.train(True)
        self.model.set_training(True)
        num_of_train_batches = len(data_loader)
        for i, data in enumerate(data_loader):
            # Every data instance is an input + label pair
            inputs, recon_inputs, labels = copy.deepcopy(data)
            inputs, recon_inputs = inputs.to(self.device),recon_inputs.to(self.device)
            y_local_onehots = [label.to(self.device) for label in labels]
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            model_inputs = inputs, y_local_onehots
            # Make predictions for this batch
            local_scores, final_outputs = self.model(model_inputs)
            
            # Compute the loss and its gradients            
            x = (local_scores,y_local_onehots,recon_inputs.transpose(1,3),final_outputs,self.model)
            global_loss,margin_loss,reconstruction_loss,l2_loss = self.criterion(x)
            # Clip gradients by global norm
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.norm_ratio)
            global_loss.backward()
            
            
           
            
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            current_global_loss += global_loss.item()
            current_margin_loss += margin_loss.item()
            current_reconstruction_loss += reconstruction_loss.item()
            current_l2_loss += l2_loss.item()
            
            last_global_loss = current_global_loss/(i+1)
            last_margin_loss = current_margin_loss/(i+1)
            last_reconstruction_loss = current_reconstruction_loss/(i+1)
            last_l2_loss = current_l2_loss/(i+1)
            progress_info = f"Training: Epoch [{epoch_index+1}], Batch [{i+1}/{num_of_train_batches}], AVGMarginLoss: {last_margin_loss}, AVGReconLoss: {last_reconstruction_loss}, AVGL2Loss: {last_l2_loss}"
            print(progress_info, end='\r')
            tb_x = epoch_index * num_of_train_batches + i + 1
            self.tb_writer.add_scalar('Training/Loss', last_loss, tb_x)
            
        print('\n')
        return  last_margin_loss+last_reconstruction_loss
    
    def validate(self,epoch_index,data_loader):
        current_global_loss = 0.
        current_margin_loss = 0.
        current_reconstruction_loss = 0.
        current_l2_loss = 0.
        
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        self.model.eval()
        self.model.set_training(False)
        eval_counter, eval_loss = 0, 0.0
        num_of_val_batches = len(data_loader)
        
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(data_loader):
                vinputs, vrecon_inputs, vlabels = copy.deepcopy(vdata)
                vinputs, vrecon_inputs = vinputs.to(self.device),vrecon_inputs.to(self.device)
                y_local_onehots = [label.to(self.device) for label in vlabels]
                # Zero your gradients for every batch!
                self.optimizer.zero_grad()
                model_inputs = vinputs, y_local_onehots
                # Make predictions for this batch
                local_scores, final_outputs = self.model(model_inputs)

                # Compute the loss and its gradients            
                x = (local_scores,y_local_onehots,vrecon_inputs.transpose(1,3),final_outputs,self.model)
                vglobal_loss,vmargin_loss,vreconstruction_loss,vl2_loss = self.criterion(x)
            
                
                current_global_loss += vglobal_loss.item()
                current_margin_loss += vmargin_loss.item()
                current_reconstruction_loss += vreconstruction_loss.item()
                current_l2_loss += vl2_loss.item()
                last_vglobal_loss = current_global_loss/(i+1)
                last_vmargin_loss = current_margin_loss/(i+1)
                last_vreconstruction_loss = current_reconstruction_loss/(i+1)
                last_vl2_loss = current_l2_loss/(i+1)       
                progress_info = f"Validation: Epoch [{epoch_index+1}], Batch [{i+1}/{num_of_val_batches}], AVGMarginLoss: {last_vmargin_loss}, AVGReconLoss: {last_vreconstruction_loss}, AVGL2Loss: {last_vl2_loss}"
                print(progress_info, end='\r')
                
                tb_x = epoch_index * num_of_val_batches + eval_counter + 1
                self.tb_writer.add_scalar('Validation/Loss',eval_loss,tb_x)
                eval_counter+=1
            print('\n')                
            return last_vmargin_loss+last_vreconstruction_loss
    
    def test(self,epoch_index,data_loader):
        print(f"Evaluating best model of epoch {epoch_index}.")
        scores_list = []
        true_onehot_labels_list = []
        self.best_model.eval()
        self.best_model.set_training(False)
        with torch.no_grad():
            for i, vdata in enumerate(data_loader):
                vinputs, vrecon_inputs, vlabels = copy.deepcopy(vdata)
                vinputs,vrecon_inputs = vinputs.to(self.device), vrecon_inputs.to(self.device)
                y_local_onehots = [label.to(self.device) for label in vlabels]
                y_global_onehots = torch.cat(y_local_onehots,dim=1)
                # Zero your gradients for every batch!
                self.optimizer.zero_grad()
                model_inputs = vinputs, y_local_onehots
                # Make predictions for this batch
                local_scores, final_outputs = self.model(model_inputs)
                global_scores = torch.cat(local_scores,dim=1)
                for j in global_scores:
                    scores_list.append(j)
                # Convert each tensor to a list of lists
                for i in y_global_onehots:
                    true_onehot_labels_list.append(i)                
        metrics_dict = dh.calc_metrics(scores_list=scores_list,labels_list=true_onehot_labels_list,topK=self.args.topK,pcp_hierarchy=self.explicit_hierarchy,pcp_threshold=self.args.pcp_threshold,num_classes_list=self.num_classes_list,device=self.device,eval_pcp=self.args.pcp_metrics_active,eval_hierarchical_metrics=True)
        # Save Metrics in Summarywriter.
        for key,value in metrics_dict.items():
            self.tb_writer.add_scalar(key,value,epoch_index)
        
        

    
    
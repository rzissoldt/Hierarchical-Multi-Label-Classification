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
class HmcNetTrainer():
    def __init__(self,model,criterion,optimizer,scheduler,training_dataset,explicit_hierarchy,num_classes_list,path_to_model,args,device=None):
        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = device
        self.best_model = None
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
        print(f'Total Classes: {sum(num_classes_list)}')
        print(f'Num Classes List: {num_classes_list}')
    
    def train_and_validate(self):
        counter = 0
        best_epoch = 0
        best_vloss = 1_000_000.
        is_fine_tuning = False
        for epoch in range(self.args.epochs):
            avg_train_loss = self.train(epoch_index=epoch)
            calc_metrics = epoch == self.args.epochs-1
            avg_val_loss = self.validate(epoch_index=epoch,calc_metrics=calc_metrics)
            self.tb_writer.flush()
            print(f'Epoch {epoch+1}: Average Train Loss {avg_train_loss}, Average Validation Loss {avg_val_loss}')
            # Decay Learningrate if Step Count is reached
            if epoch % self.args.decay_steps == self.args.decay_steps-1:
                self.scheduler.step()
            # Track best performance, and save the model's state
            if avg_val_loss < best_vloss:
                best_epoch = epoch
                self.best_model = copy.deepcopy(self.model)
                best_vloss = avg_val_loss
                model_path = os.path.join(self.path_to_model,'models',f'hmcnet_{epoch+1}')
                counter = 0
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(self.model.state_dict(), model_path)
            else:
                counter += 1
                if counter >= self.args.early_stopping_patience and not is_fine_tuning:
                    print(f'Early stopping triggered and validate best Epoch {best_epoch+1}.')
                    print(f'Begin fine tuning model.')
                    avg_val_loss = self.validate(epoch_index=epoch,calc_metrics=True)
                    self.unfreeze_backbone()
                    best_vloss = 1_000_000.
                    is_fine_tuning = True
                    counter = 0
                    continue
                if counter >= self.args.early_stopping_patience and is_fine_tuning:
                    print(f'Early stopping triggered in fine tuning Phase. {best_epoch+1} was the best Epoch.')
                    print(f'Validate fine tuned Model.')
                    avg_val_loss = self.validate(epoch_index=epoch,calc_metrics=True)
                    break
    
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
                avg_val_loss = self.validate(epoch_index=epoch, data_loader=val_loader, calc_metrics=calc_metrics)
                self.tb_writer.flush()
                print(f'Epoch {epoch+1}: Average Train Loss {avg_train_loss}, Average Validation Loss {avg_val_loss}')
                epoch+=1
                # Decay Learning rate if Step Count is reached
                if epoch % self.args.decay_steps == self.args.decay_steps - 1:
                    self.scheduler.step()

                # Track best performance, and save the model's state
                if avg_val_loss < best_vloss:
                    best_epoch = epoch
                    self.best_model = copy.deepcopy(self.model)
                    best_vloss = avg_val_loss
                    model_path = os.path.join(self.path_to_model, 'models', f'hmcnet_{epoch+1}')
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    torch.save(self.model.state_dict(), model_path)
                else:
                    counter += 1
                    if counter >= self.args.early_stopping_patience and not is_fine_tuning:
                        print(f'Early stopping triggered and validate best Epoch {best_epoch + 1}.')
                        print(f'Begin fine-tuning model.')
                        avg_val_loss = self.validate(epoch_index=epoch, data_loader=val_loader, calc_metrics=True)
                        self.unfreeze_backbone()
                        best_vloss = 1_000_000.
                        is_fine_tuning = True
                        counter = 0
                        continue
                    if counter >= self.args.early_stopping_patience and is_fine_tuning:
                        print(f'Early stopping triggered in fine-tuning Phase. {best_epoch + 1} was the best Epoch.')
                        print(f'Validate fine-tuned Model.')
                        avg_val_loss = self.validate(epoch_index=epoch, data_loader=val_loader, calc_metrics=True)
                        is_finished = True
                        break
                    
    def train(self,epoch_index):
        current_loss = 0.
        current_global_loss = 0.
        current_local_loss = 0.
        current_hierarchy_loss = 0.
        current_l2_loss = 0.
        last_loss = 0.
        self.model.train(True)
        num_of_train_batches = len(self.training_loader)
        for i, data in enumerate(self.training_loader):
            # Every data instance is an input + label pair
            inputs, labels = copy.deepcopy(data)
            inputs = inputs.to(self.device)
            y_total_onehot = labels[0].to(self.device)
            y_local_onehots = [label.to(self.device) for label in labels[1:]]
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            _, local_scores_list, global_logits = self.model(inputs)

            # Compute the loss and its gradients
            predictions = (local_scores_list,global_logits)
            targets = (y_local_onehots,y_total_onehot)
            x = (predictions,targets,self.model)
            loss,global_loss,local_loss,hierarchy_loss,l2_loss = self.criterion(x)
            loss.backward()
            
            # Clip gradients by global norm
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.norm_ratio)
            
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            current_loss += loss.item()
            current_global_loss += global_loss.item()
            current_local_loss += local_loss.item()
            current_hierarchy_loss += hierarchy_loss.item()
            current_l2_loss += l2_loss.item()
            last_loss = current_loss/(i+1)
            last_global_loss = current_global_loss/(i+1)
            last_local_loss = current_local_loss/(i+1)
            last_hierarchy_loss = current_hierarchy_loss/(i+1)
            last_l2_loss = current_l2_loss/(i+1)
            progress_info = f"Training: Epoch [{epoch_index+1}], Batch [{i+1}/{num_of_train_batches}], AVGLoss: {last_global_loss+last_local_loss+last_hierarchy_loss}, L2Loss: {last_l2_loss}"
            print(progress_info, end='\r')
            tb_x = epoch_index * num_of_train_batches + i + 1
            self.tb_writer.add_scalar('Training/Loss', last_loss, tb_x)
            self.tb_writer.add_scalar('Training/GlobalLoss', last_global_loss, tb_x)
            self.tb_writer.add_scalar('Training/LocalLoss', last_local_loss, tb_x)
            self.tb_writer.add_scalar('Training/HierarchyLoss', last_hierarchy_loss, tb_x)
            self.tb_writer.add_scalar('Training/L2Loss', last_l2_loss, tb_x)
        print('\n')
        return last_global_loss+last_local_loss+last_hierarchy_loss
    
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
            _, local_scores_list, global_logits = self.model(inputs)

            # Compute the loss and its gradients
            predictions = (local_scores_list,global_logits)
            targets = (y_local_onehots,y_total_onehot)
            x = (predictions,targets,self.model)
            loss,global_loss,local_loss,hierarchy_loss,l2_loss = self.criterion(x)
            loss.backward()
            
            # Clip gradients by global norm
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.norm_ratio)
            
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            current_loss += loss.item()
            current_global_loss += global_loss.item()
            current_local_loss += local_loss.item()
            current_hierarchy_loss += hierarchy_loss.item()
            current_l2_loss += l2_loss.item()
            last_loss = current_loss/(i+1)
            last_global_loss = current_global_loss/(i+1)
            last_local_loss = current_local_loss/(i+1)
            last_hierarchy_loss = current_hierarchy_loss/(i+1)
            last_l2_loss = current_l2_loss/(i+1)
            progress_info = f"Training: Epoch [{epoch_index+1}], Batch [{i+1}/{num_of_train_batches}], AVGLoss: {last_global_loss+last_local_loss+last_hierarchy_loss}, L2Loss: {last_l2_loss}"
            print(progress_info, end='\r')
            tb_x = epoch_index * num_of_train_batches + i + 1
            self.tb_writer.add_scalar('Training/Loss', last_loss, tb_x)
            self.tb_writer.add_scalar('Training/GlobalLoss', last_global_loss, tb_x)
            self.tb_writer.add_scalar('Training/LocalLoss', last_local_loss, tb_x)
            self.tb_writer.add_scalar('Training/HierarchyLoss', last_hierarchy_loss, tb_x)
            self.tb_writer.add_scalar('Training/L2Loss', last_l2_loss, tb_x)
        print('\n')
        return last_global_loss+last_local_loss+last_hierarchy_loss
    
    def validate(self,epoch_index,data_loader,calc_metrics=False):
        running_vloss = 0.0
        if calc_metrics:
            self.model = copy.deepcopy(self.best_model)
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
                x = (predictions,targets,self.model)
                vloss,vglobal_loss,vlocal_loss,vhierarchy_loss,vl2_loss = self.criterion(x)
                current_vglobal_loss += vglobal_loss.item()
                current_vlocal_loss += vlocal_loss.item()
                current_vhierarchy_loss += vhierarchy_loss.item()
                current_vl2_loss += vl2_loss.item()
                last_vglobal_loss = current_vglobal_loss/(i+1)
                last_vlocal_loss = current_vlocal_loss/(i+1)
                last_vhierarchy_loss = current_vhierarchy_loss/(i+1)
                last_vl2_loss = current_vl2_loss/(i+1)
                running_vloss += vloss.item()
                for j in scores:
                    scores_list.append(j)
                # Convert each tensor to a list of lists
                for i in y_total_onehot:
                    true_onehot_labels_list.append(i)                
                eval_loss = running_vloss/(eval_counter+1)
                #progress_info = f'Validation: Epoch [{epoch_index+1}], Batch [{eval_counter+1}/{num_of_val_batches}], AVGLoss: {eval_loss}'
                progress_info = f"Validation: Epoch [{epoch_index+1}], Batch [{eval_counter+1}/{num_of_val_batches}], AVGLoss: {last_vglobal_loss+last_vlocal_loss+last_vhierarchy_loss}, L2Loss: {last_vl2_loss}"
                print(progress_info, end='\r')
                if not calc_metrics:
                    tb_x = epoch_index * num_of_val_batches + eval_counter + 1
                    self.tb_writer.add_scalar('Validation/Loss',eval_loss,tb_x)
                    self.tb_writer.add_scalar('Validation/GlobalLoss',last_vglobal_loss,tb_x)
                    self.tb_writer.add_scalar('Validation/LocalLoss',last_vlocal_loss,tb_x)
                    self.tb_writer.add_scalar('Validation/HierarchyLoss',last_vhierarchy_loss,tb_x)
                    self.tb_writer.add_scalar('Validation/L2Loss',last_vl2_loss,tb_x)
                eval_counter+=1
            print('\n')
            if calc_metrics:
                # Predict classes by threshold or topk ('ts': threshold; 'tk': topk)
                eval_pre_pcp_tk = [0.0] * self.args.topK
                eval_rec_pcp_tk = [0.0] * self.args.topK
                eval_F1_pcp_tk = [0.0] * self.args.topK
                predicted_pcp_onehot_labels_ts = []
                predicted_pcp_onehot_labels_tk = [[] for _ in range(self.args.topK)]
                
                scores = torch.cat([torch.unsqueeze(tensor,0) for tensor in scores_list],dim=0)
                
                # Predict by pcp-threshold
                batch_predicted_onehot_labels_ts = dh.get_pcp_onehot_label_threshold(scores=scores,explicit_hierarchy=self.explicit_hierarchy,num_classes_list=self.num_classes_list, pcp_threshold=self.args.pcp_threshold)
                for k in batch_predicted_onehot_labels_ts:
                    predicted_pcp_onehot_labels_ts.append(k)
                
                # Predict by pcp-topK
                for top_num in range(self.args.topK):
                    batch_predicted_pcp_onehot_labels_tk = dh.get_pcp_onehot_label_topk(scores=scores,explicit_hierarchy=self.explicit_hierarchy,pcp_threshold=self.args.pcp_threshold,num_classes_list=self.num_classes_list, top_num=top_num+1)
                    for i in batch_predicted_pcp_onehot_labels_tk:
                        predicted_pcp_onehot_labels_tk[top_num].append(i)
                        
                # Calculate Precision & Recall & F1
                predicted_pcp_onehot_labels = torch.cat([torch.unsqueeze(tensor,0) for tensor in predicted_pcp_onehot_labels_ts],dim=0).to(self.device)

                true_onehot_labels = torch.cat([torch.unsqueeze(tensor,0) for tensor in true_onehot_labels_list],dim=0).to(self.device)

                eval_pre_pcp_ts,eval_rec_pcp_ts,eval_F1_pcp_ts = dh.precision_recall_f1_score(labels=true_onehot_labels,binary_predictions=predicted_pcp_onehot_labels, average='micro')


                for top_num in range(self.args.topK):
                    predicted_pcp_onehot_labels_topk = torch.cat([torch.unsqueeze(tensor,0) for tensor in predicted_pcp_onehot_labels_tk[top_num]],dim=0).to(self.device)
                    eval_pre_pcp_tk[top_num], eval_rec_pcp_tk[top_num],eval_F1_pcp_tk[top_num] = dh.precision_recall_f1_score(labels=true_onehot_labels,binary_predictions=predicted_pcp_onehot_labels_topk, average='micro')

                num_total_classes = predicted_pcp_onehot_labels.shape[1]
                auroc = AUROC(task="binary")
                eval_auc = auroc(predicted_pcp_onehot_labels,true_onehot_labels.to(dtype=torch.long))
                auprc = AveragePrecision(task="binary")
                eval_auprc = auprc(predicted_pcp_onehot_labels,true_onehot_labels.to(dtype=torch.long))
                eval_loss = running_vloss/(eval_counter+1)
                
                self.tb_writer.add_scalar('Validation/AverageAUC',eval_auc,epoch_index)
                self.tb_writer.add_scalar('Validation/AveragePrecision',eval_auprc,epoch_index)
                self.tb_writer.add_scalar('Validation/PCPPrecision',eval_pre_pcp_ts,epoch_index)
                self.tb_writer.add_scalar('Validation/PCPRecall',eval_rec_pcp_ts,epoch_index)
                self.tb_writer.add_scalar('Validation/PCPF1',eval_F1_pcp_ts,epoch_index)
                for i, precision in enumerate(eval_pre_pcp_tk):
                    self.tb_writer.add_scalar(f'Validation/PCPPrecisionTopK/{i}', precision, global_step=epoch_index)
                for i, recall in enumerate(eval_rec_pcp_tk):
                    self.tb_writer.add_scalar(f'Validation/PCPRecallTopK/{i}', recall, global_step=epoch_index)
                for i, f1 in enumerate(eval_F1_pcp_tk):
                    self.tb_writer.add_scalar(f'Validation/PCPF1TopK/{i}', f1, global_step=epoch_index)

                print("All Validation set: Loss {0:g} | AUC {1:g} | AUPRC {2:g}".format(eval_loss, eval_auc, eval_auprc))
                # Predict by pcp
                print("Predict by PCP thresholding: PCP-Precision {0:g}, PCP-Recall {1:g}, PCP-F1 {2:g}".format(eval_pre_pcp_ts, eval_rec_pcp_ts, eval_F1_pcp_ts))
                # Predict by PCP-topK
                print("Predict by PCP-topK:")
                for top_num in range(self.args.topK):
                    print("Top{0}: PCP-Precision {1:g}, PCP-Recall {2:g}, PCP-F1 {3:g}".format(top_num+1, eval_pre_pcp_tk[top_num], eval_rec_pcp_tk[top_num], eval_F1_pcp_tk[top_num]))  
            return last_vglobal_loss+last_vlocal_loss+last_vhierarchy_loss
        
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
        param_groups[2]['params'] = self.model.ham_modules.parameters()
        param_groups[2]['lr'] = base_lr
        param_groups[3]['params'] = self.model.hybrid_predicting_module.parameters()
        param_groups[3]['lr'] = base_lr

        # Update the optimizer with the new parameter groups
        self.optimizer.param_groups = param_groups
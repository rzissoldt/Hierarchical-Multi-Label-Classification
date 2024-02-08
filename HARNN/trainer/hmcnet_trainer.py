import copy
import torch
import sys
import numpy as np
sys.path.append('../')
from utils import data_helpers as dh
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
class HmcNetTrainer():
    def __init__(self,model,criterion,optimizer,scheduler,explicit_hierarchy,num_classes_list,args,device=None):
        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = device
        self.explicit_hierarchy= explicit_hierarchy
        self.args = args
        self.num_classes_list = num_classes_list
    def train(self,training_loader,epoch_index,tb_writer):
        current_loss = 0.
        last_loss = 0.
        self.model.train(True)
        num_of_train_batches = len(training_loader)
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = copy.deepcopy(data)
            inputs = inputs.to(self.device)
            y_total_onehot = labels[0]
            y_local_onehots = labels[1:]
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            scores, local_scores_list, global_logits = self.model(inputs)

            # Compute the loss and its gradients
            predictions = (local_scores_list,global_logits)
            targets = (y_local_onehots,y_total_onehot)
            loss = self.criterion(predictions=predictions,targets=targets)
            loss.backward()
            
            # Clip gradients by global norm
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.norm_ratio)
            
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            current_loss += loss.item()
            last_loss = current_loss/(i+1)

            progress_info = f"Training: Epoch [{epoch_index+1}], Batch [{i+1}/{num_of_train_batches}], AVGLoss: {last_loss}, Loss: {current_loss}"
            print(progress_info, end='\r')
            tb_x = epoch_index * num_of_train_batches + i + 1
            tb_writer.add_scalar('Training/Loss', last_loss, tb_x)
        print('\n')
        return last_loss
    
    def validate(self,validation_loader,epoch_index,tb_writer):
        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        self.model.eval()

        # Predict classes by threshold or topk ('ts': threshold; 'tk': topk)
        eval_counter, eval_loss = 0, 0.0
        eval_pre_tk = [0.0] * self.args.topK
        eval_rec_tk = [0.0] * self.args.topK
        eval_F1_tk = [0.0] * self.args.topK
        eval_pre_pcp_tk = [0.0] * self.args.topK
        eval_rec_pcp_tk = [0.0] * self.args.topK
        eval_F1_pcp_tk = [0.0] * self.args.topK
        true_onehot_labels = []
        predicted_onehot_scores = []
        predicted_pcp_onehot_labels_ts = []
        predicted_onehot_labels_tk = [[] for _ in range(self.args.topK)]
        predicted_pcp_onehot_labels_tk = [[] for _ in range(self.args.topK)]
        num_of_val_batches = len(validation_loader)
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = copy.deepcopy(vdata)
                vinputs = vinputs.to(self.device)
                y_total_onehot = vlabels[0]
                y_local_onehots = vlabels[1:]
                # Make predictions for this batch
                scores, local_scores_list, global_logits = self.model(vinputs)

                # Compute the loss and its gradients
                predictions, targets = (local_scores_list,global_logits),(y_local_onehots,y_total_onehot)
                vloss = self.criterion(predictions=predictions,targets=targets)
                
                scores = scores.cpu().numpy()
                running_vloss += vloss.item()
                # Convert each tensor to a list of lists
                y_total_onehot_list = [total_onehot.tolist() for total_onehot in list(torch.stack(y_total_onehot).t())]
                # Prepare for calculating metrics
                for i in y_total_onehot_list:
                    true_onehot_labels.append(i)
                for j in scores:
                    predicted_onehot_scores.append(j)
                # Predict by pcp-threshold
                batch_predicted_onehot_labels_ts = \
                    dh.get_pcp_onehot_label_threshold(scores=scores,explicit_hierarchy=self.explicit_hierarchy,num_classes_list=self.num_classes_list, pcp_threshold=self.args.pcp_threshold)
                for k in batch_predicted_onehot_labels_ts:
                    predicted_pcp_onehot_labels_ts.append(k)
                # Predict by topK
                for top_num in range(self.args.topK):
                    batch_predicted_onehot_labels_tk = dh.get_onehot_label_topk(scores=scores, top_num=top_num+1)
                    for i in batch_predicted_onehot_labels_tk:
                        predicted_onehot_labels_tk[top_num].append(i)
                # Predict by pcp-topK
                for top_num in range(self.args.topK):
                    batch_predicted_pcp_onehot_labels_tk = dh.get_pcp_onehot_label_topk(scores=scores,explicit_hierarchy=self.explicit_hierarchy,pcp_threshold=self.args.pcp_threshold,num_classes_list=self.num_classes_list, top_num=top_num+1)
                    for i in batch_predicted_pcp_onehot_labels_tk:
                        predicted_pcp_onehot_labels_tk[top_num].append(i)
                
                eval_loss = running_vloss/(eval_counter+1)
                progress_info = f"Validation: Epoch [{epoch_index+1}], Batch [{eval_counter+1}/{num_of_val_batches}], AVGLoss: {eval_loss}, Loss: {running_vloss}"
                print(progress_info, end='\r')
                eval_counter+=1
            print('\n')
            # Calculate Precision & Recall & F1
            eval_pre_pcp_ts = precision_score(y_true=np.array(true_onehot_labels),
                                          y_pred=np.array(predicted_pcp_onehot_labels_ts), average='micro')
            eval_rec_pcp_ts = recall_score(y_true=np.array(true_onehot_labels),
                                       y_pred=np.array(predicted_pcp_onehot_labels_ts), average='micro')
            eval_F1_pcp_ts = f1_score(y_true=np.array(true_onehot_labels),
                                  y_pred=np.array(predicted_pcp_onehot_labels_ts), average='micro')
            for top_num in range(self.args.topK):
                eval_pre_tk[top_num] = precision_score(y_true=np.array(true_onehot_labels),
                                                       y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                       average='micro')
                eval_rec_tk[top_num] = recall_score(y_true=np.array(true_onehot_labels),
                                                    y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                    average='micro')
                eval_F1_tk[top_num] = f1_score(y_true=np.array(true_onehot_labels),
                                               y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                               average='micro')
            for top_num in range(self.args.topK):
                eval_pre_pcp_tk[top_num] = precision_score(y_true=np.array(true_onehot_labels),
                                                       y_pred=np.array(predicted_pcp_onehot_labels_tk[top_num]),
                                                       average='micro')
                eval_rec_pcp_tk[top_num] = recall_score(y_true=np.array(true_onehot_labels),
                                                    y_pred=np.array(predicted_pcp_onehot_labels_tk[top_num]),
                                                    average='micro')
                eval_F1_pcp_tk[top_num] = f1_score(y_true=np.array(true_onehot_labels),
                                               y_pred=np.array(predicted_pcp_onehot_labels_tk[top_num]),
                                               average='micro')
            
            eval_loss = running_vloss/(eval_counter+1)
            # Calculate the average AUC
            eval_auc = roc_auc_score(y_true=np.array(true_onehot_labels),
                                     y_score=np.array(predicted_onehot_scores), average='micro')
            # Calculate the average PR
            eval_prc = average_precision_score(y_true=np.array(true_onehot_labels),
                                               y_score=np.array(predicted_onehot_scores), average='micro')
            tb_writer.add_scalar('Validation/Loss',eval_loss,epoch_index)
            tb_writer.add_scalar('Validation/AverageAUC',eval_auc,epoch_index)
            tb_writer.add_scalar('Validation/AveragePrecision',eval_prc,epoch_index)
            # Add each scalar individually
            for i, precision in enumerate(eval_pre_tk):
                tb_writer.add_scalar(f'Validation/PrecisionTopK/{i}', precision, global_step=epoch_index)
            for i, recall in enumerate(eval_rec_tk):
                tb_writer.add_scalar(f'Validation/RecallTopK/{i}', recall, global_step=epoch_index)
            for i, f1 in enumerate(eval_F1_tk):
                tb_writer.add_scalar(f'Validation/F1TopK/{i}', f1, global_step=epoch_index)
            
            tb_writer.add_scalar('Validation/PCPPrecision',eval_pre_pcp_ts,epoch_index)
            tb_writer.add_scalar('Validation/PCPRecall',eval_rec_pcp_ts,epoch_index)
            tb_writer.add_scalar('Validation/PCPF1',eval_F1_pcp_ts,epoch_index)
            for i, precision in enumerate(eval_pre_pcp_tk):
                tb_writer.add_scalar(f'Validation/PCPPrecisionTopK/{i}', precision, global_step=epoch_index)
            for i, recall in enumerate(eval_rec_pcp_tk):
                tb_writer.add_scalar(f'Validation/PCPRecallTopK/{i}', recall, global_step=epoch_index)
            for i, f1 in enumerate(eval_F1_pcp_tk):
                tb_writer.add_scalar(f'Validation/PCPF1TopK/{i}', f1, global_step=epoch_index)
        
            print("All Validation set: Loss {0:g} | AUC {1:g} | AUPRC {2:g}".format(eval_loss, eval_auc, eval_prc))
            # Predict by pcp
            print("Predict by PCP thresholding: PCP-Precision {0:g}, PCP-Recall {1:g}, PCP-F1 {2:g}".format(eval_pre_pcp_ts, eval_rec_pcp_ts, eval_F1_pcp_ts))
            # Predict by topK
            print("Predict by topK:")
            for top_num in range(self.args.topK):
                print("Top{0}: Precision {1:g}, Recall {2:g}, F1 {3:g}".format(top_num+1, eval_pre_tk[top_num], eval_rec_tk[top_num], eval_F1_tk[top_num])) 
            # Predict by PCP-topK
            print("Predict by PCP-topK:")
            for top_num in range(self.args.topK):
                print("Top{0}: PCP-Precision {1:g}, PCP-Recall {2:g}, PCP-F1 {3:g}".format(top_num+1, eval_pre_pcp_tk[top_num], eval_rec_pcp_tk[top_num], eval_F1_pcp_tk[top_num]))  
            return eval_loss
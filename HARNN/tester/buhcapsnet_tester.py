import copy
import torch
import sys

sys.path.append('../')
from utils import data_helpers as dh

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class BUHCapsNetTester():
    def __init__(self,model,num_classes_list,test_dataset,sample_images_size,explicit_hierarchy,path_to_results,hierarchy_dicts,args,device=None):
        self.model = model
        self.device = device
        self.best_model = copy.deepcopy(model)
        self.hierarchy_dicts = hierarchy_dicts
        self.explicit_hierarchy = explicit_hierarchy
        self.args = args
        self.total_class_num = sum(num_classes_list)
        self.num_classes_list = num_classes_list   
        self.path_to_results = path_to_results
        self.tb_writer = SummaryWriter(path_to_results)
        sharing_strategy = "file_system"
        def set_worker_sharing_strategy(worker_id: int):
            torch.multiprocessing.set_sharing_strategy(sharing_strategy)
        # Create Dataloader for Training and Validation Dataset
        kwargs = {'num_workers': args.num_workers_dataloader, 'pin_memory': args.pin_memory} if self.args.gpu else {}
        self.test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True,worker_init_fn=set_worker_sharing_strategy,**kwargs) 
        
    
    def test(self,epoch_index,data_loader):
        print(f"Evaluating best model of epoch {epoch_index}.")
        scores_list = []
        true_onehot_labels_list = []
        self.best_model.eval()
        self.best_model.set_training(False)
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
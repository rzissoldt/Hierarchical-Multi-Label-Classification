import copy
import torch
import sys
sys.path.append('../')
from utils import data_helpers as dh
import numpy as np
import os

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class HmcNetTester():
    def __init__(self,model,num_classes_list,test_dataset,sample_images_size,explicit_hierarchy,path_to_results,args,device=None):
        self.model = model
        self.device = device
        self.best_model = copy.deepcopy(model)
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
        total_samples = len(self.test_loader.dataset)

        # Generate random indices for the subset
        random_seed = 42
        np.random.seed(random_seed)
        random_indices = np.random.choice(total_samples, size=sample_images_size, replace=False)
        subset_test_dataset = torch.utils.data.Subset(self.test_loader.dataset, random_indices)
        self.subset_test_loader = DataLoader(subset_test_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=set_worker_sharing_strategy, **kwargs)
    def test(self):
        print(f"Evaluating best model with test dataset.")
        scores_list = []
        labels_list = []
        self.best_model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(self.test_loader):
                vinputs, vlabels = copy.deepcopy(vdata)
                vinputs = vinputs.to(self.device)
                y_total_onehot = vlabels[0].to(self.device)
                y_local_onehots = [label.to(self.device) for label in vlabels[1:]]
                # Make predictions for this batch
                scores, local_scores_list, global_logits = self.best_model(vinputs)
                scores_list.extend(scores)
                labels_list.extend(y_total_onehot)              
        metrics_dict = dh.calc_metrics(scores_list=scores_list,labels_list=labels_list,topK=self.args.topK,pcp_hierarchy=self.explicit_hierarchy.to('cpu').numpy(),pcp_threshold=self.args.pcp_threshold,num_classes_list=self.num_classes_list,device=self.device,eval_pcp=self.args.pcp_metrics_active,threshold=self.args.threshold,eval_hierarchical_metrics=True)
        # Save Metrics in Summarywriter.
        for key,value in metrics_dict.items():
            self.tb_writer.add_scalar(key,value,0)
        
         # Visualization of results
        #output_file_path = os.path.join(self.path_to_results,'sample_images')
        #os.makedirs(output_file_path, exist_ok=True)
        #image_list = []
        #label_list = []
        #score_list = []
        #with torch.no_grad():
        #    for i, vdata in enumerate(self.subset_test_loader):
        #        vinputs, vlabels = copy.deepcopy(vdata)
        #        vinputs = vinputs.to(self.device)
        #        y_total_onehot = vlabels[0].to(self.device)
        #        y_local_onehots = [label.to(self.device) for label in vlabels[1:]]
        #        # Make predictions for this batch
        #        scores, local_scores_list, global_logits = self.best_model(vinputs)
        #        for input in vinputs:
        #            image_list.append(input)
        #        for label in vlabels:
        #            label_list.append(label)
        #        for score in scores:
        #            score_list.append(score)
        #dh.visualize_sample_images(images=image_list,true_labels=label_list,scores=score_list,threshold=self.args.threshold,hierarchy_dicts=self.hierarchy_dicts,output_file_path=output_file_path)
import copy
import torch
import sys

sys.path.append('../')
from utils import data_helpers as dh

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class BaselineTester():
    def __init__(self,model,num_classes_list,test_dataset,explicit_hierarchy,path_to_results,args,device=None):
        self.model = model
        self.device = device
        self.best_model = copy.deepcopy(model)
        self.explicit_hierarchy = explicit_hierarchy
        self.args = args
        self.total_class_num = sum(num_classes_list)
        self.num_classes_list = num_classes_list   
        self.tb_writer = SummaryWriter(path_to_results)
        sharing_strategy = "file_system"
        def set_worker_sharing_strategy(worker_id: int):
            torch.multiprocessing.set_sharing_strategy(sharing_strategy)
        # Create Dataloader for Training and Validation Dataset
        kwargs = {'num_workers': args.num_workers_dataloader, 'pin_memory': args.pin_memory} if self.args.gpu else {}
        self.test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True,worker_init_fn=set_worker_sharing_strategy,**kwargs) 
    
    def test(self):
        print(f"Evaluating best model with test dataset.")
        self.best_model.eval()
        scores_list = []
        labels_list = []
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(self.test_loader):
                # Every data instance is an input + label pair
                inputs, labels = copy.deepcopy(vdata)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

               # Make predictions for this batch
                output = self.best_model(inputs.float())
                scores_list.extend(output)
                labels_list.extend(labels) 
        metrics_dict = dh.calc_metrics(scores_list=scores_list,labels_list=labels_list,topK=self.args.topK,pcp_hierarchy=self.explicit_hierarchy.to('cpu').numpy(),pcp_threshold=self.args.pcp_threshold,num_classes_list=self.num_classes_list,device=self.device,eval_pcp=True,threshold=self.args.threshold)
        # Save Metrics in Summarywriter.
        for key,value in metrics_dict.items():
            self.tb_writer.add_scalar(key,value,0)
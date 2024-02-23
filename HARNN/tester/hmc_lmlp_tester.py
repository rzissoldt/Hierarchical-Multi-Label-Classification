import copy
import torch
import sys
sys.path.append('../')
from utils import data_helpers as dh

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from HARNN.model.chmcnn_model import get_constr_out
from sklearn.metrics import average_precision_score
class HmcLMLPTester():
    def __init__(self,model,test_dataset,num_classes_list,path_to_results,pcp_hierarchy,args,device=None):
        self.model = model
        self.device = device
        self.pcp_hierarchy = pcp_hierarchy
        self.best_model = copy.deepcopy(model)
        self.args = args
        self.num_classes_list = num_classes_list
        self.path_to_results = path_to_results        
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
                inputs = inputs, len(self.num_classes_list)-1
                _, output_scores_list = self.best_model(inputs)
                

                # Stack the batch tensors along a new dimension (dimension 0)
                
                scores = torch.cat([batch for batch in output_scores_list],dim=1)
                scores_list.extend([score.to(dtype=torch.float64) for score in scores])
                labels_list.extend(labels)
            
        metrics_dict = dh.calc_metrics(scores_list=scores_list,labels_list=labels_list,topK=self.args.topK,pcp_hierarchy=self.pcp_hierarchy,pcp_threshold=self.args.pcp_threshold,num_classes_list=self.num_classes_list,device=self.device)
        # Save Metrics in Summarywriter.
        for key,value in metrics_dict.items():
            self.tb_writer.add_scalar(key,value,0)
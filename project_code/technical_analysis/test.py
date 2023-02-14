import os
import sys
path_dir = os.path.abspath(os.getcwd())
sys.path.append(path_dir+"/project_code")
data_dir = path_dir + "/project_code/technical_analysis/data/"
save_dir = path_dir + "/project_code/technical_analysis/outputs/"
model_dir = path_dir + "/project_code/technical_analysis/models/"

from technical_analysis.models import MLP
from technical_analysis.train_util import dataset
import torch
from torch.utils.data import DataLoader
import wandb
import numpy as np

import argparse
parser = argparse.ArgumentParser()
#feature extractor
parser.add_argument("--feature_extractor",type=str,default="mlp",required=True)
#device
parser.add_argument("--device",type=str,default="cuda",required=True)
#batch_size
parser.add_argument("--batch_size",type=int,default=32,required=True)
#episodes
parser.add_argument("--epochs",type=int,default=1000,required=True)
#lr
parser.add_argument("--lr",type=float,default=1e-4,required=True)
args = parser.parse_args()


# data
test = dataset(path_x=data_dir+"x_test.npy",path_gt=data_dir+"gt_test.npy")
# data loader
test_loader=DataLoader(dataset = test,batch_size=1,shuffle=True)

# model
if  args.feature_extractor == "mlp":
    model = MLP(in_dim=15,hidden_dim=10,out_dim=2).to(device=args.device).double()
    model.load_state_dict(torch.load(model_dir+"mlp_lr-{}_batch_size-{}_epochs-{}.pth".format(args.lr,args.batch_size,args.epochs)))
elif args.feature_extractor == "lstm":
    pass
elif args.feature_extractor == "transformer":
    pass

def test():
    # wandb.init(project="TECHNICHAL_ANALYSIS",name="mlp_lr-{}_batch_size-{}_epoch-{}".format(args.lr,args.batch_size,args.epochs),entity="g0kul6")
    outputs = []
    gts = []
    accuracy = 0
    loss_function = torch.nn.CrossEntropyLoss()
    for data,label in test_loader:
        data = data.cuda()
        label = label.cuda()
        output = model(data,b_s=args.batch_size)
        gts.append(label.cpu().item())
        outputs.append(output.argmax(dim=1).cpu().item())
        loss = loss_function(output,label)
        acc = (output.argmax(dim=1) == label).float().mean()
        accuracy = accuracy + acc.cpu().item()
    print("Total_accuracy:",accuracy)
    return outputs,gts

if __name__ == '__main__':
    output,gt = test()
    np.save(save_dir+"output_mlp_lr-{}_batch_size-{}_epoch-{}".format(args.lr,args.batch_size,args.epochs),output)
    np.save(save_dir+"gt_mlp_lr-{}_batch_size-{}_epoch-{}".format(args.lr,args.batch_size,args.epochs),gt)
    

        
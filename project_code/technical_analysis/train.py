import os
import sys
path_dir = os.path.abspath(os.getcwd())
sys.path.append(path_dir+"/project_code")
data_dir = path_dir + "/project_code/technical_analysis/data/"


from torch.utils.data import DataLoader
from technical_analysis.train_util import dataset,train_mlp
import wandb

import argparse
parser = argparse.ArgumentParser()
#feature extractor
parser.add_argument("--feature_extractor",type=str,default="mlp",required=True)
#device
parser.add_argument("--device",type=str,default="cuda",required=True)
#episodes
parser.add_argument("--epochs",type=int,default=1000,required=True)
#lr
parser.add_argument("--lr",type=float,default=1e-4,required=True)
#batch_size
parser.add_argument("--batch_size",type=int,default=32,required=True)
args = parser.parse_args()

# loading train and test data
train = dataset(path_x=data_dir+"x_train.npy",path_gt=data_dir+"gt_train.npy")
val = dataset(path_x=data_dir+"x_val.npy",path_gt=data_dir+"gt_val.npy")
# data loader 
train_loader=DataLoader(dataset = train,batch_size=args.batch_size,shuffle=True)
val_loader = DataLoader(dataset = val,batch_size=args.batch_size,shuffle=True)


if args.feature_extractor == "mlp":
    train_mlp(input_dim=15,hidden_dim=10,output_dim=2,lr=args.lr,device=args.device,
              train=train_loader,val=val_loader,epochs=args.epochs,batch_size=args.batch_size)
elif  args.feature_extractor == "lstm":
    pass
elif  args.feature_extractor == "transformer":
    pass
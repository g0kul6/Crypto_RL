import os
import sys
path_dir = os.path.abspath(os.getcwd())
sys.path.append(path_dir+"/project_code")
import wandb
model_dir = path_dir + "/project_code/technical_analysis/models/"

import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from technical_analysis.models import MLP
import torch.optim as optim

class dataset(Dataset):
  def __init__(self,path_x,path_gt):
    self.x = np.load(path_x)
    self.gt = np.load(path_gt)
    self.path_x = path_x
    self.path_gt = path_gt
  def __len__(self):
    self.filelength=len(self.gt)
    return self.filelength
  def __getitem__(self,idx):
    x_i = self.x[idx]
    gt_i = self.gt[idx]
    return torch.tensor(x_i),torch.tensor(gt_i)

def train_mlp(input_dim,hidden_dim,output_dim,lr,device,train,val,epochs,batch_size):
  wandb.init(project="TECHNICHAL_ANALYSIS",name="mlp_lr-{}_batch_size-{}_epoch-{}".format(lr,batch_size,epochs),entity="g0kul6")
  model = MLP(in_dim=input_dim,hidden_dim=hidden_dim,out_dim=output_dim).to(device=device).double()
  optimizer = optim.Adam(model.parameters(),lr=lr)
  loss_function = torch.nn.CrossEntropyLoss()
  for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for data,label in train:
      data = data.cuda()
      label = label.cuda()
      output = model(data,b_s=batch_size)
      loss = loss_function(output,label)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      acc = (output.argmax(dim=1) == label).float().mean()
      epoch_accuracy += acc/len(train)
      epoch_loss += loss/len(train)
    with torch.no_grad():
        epoch_val_accuracy=0
        epoch_val_loss =0
        for data, label in val:
            data = data.to(device)
            label = label.to(device)
            val_output = model(data,b_s=batch_size)
            val_loss = loss_function(val_output,label)
            acc = ((val_output.argmax(dim=1) == label).float().mean())
            epoch_val_accuracy += acc/ len(val)
            epoch_val_loss += val_loss/ len(val)
    wandb.log({"train_loss":epoch_loss,"train_accuracy":epoch_accuracy,"val_loss":epoch_val_loss,"val_accuracy":epoch_val_accuracy})
    print("Episode:",epoch,"Loss_Train:",epoch_loss.item(),"Accuracy_Train:",epoch_accuracy.item(),"Loss_Val",epoch_val_loss.item(),"Accuracy_Val",epoch_val_accuracy.item())
    torch.save(model.state_dict(),model_dir+"mlp_lr-{}_batch_size-{}_epochs-{}.pth".format(lr,batch_size,epochs))






    
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
writer = SummaryWriter('results/test/test_tb')

n=10
m=10
k=10
net=nn.Sequential(nn.Linear(n,m),nn.ReLU(),nn.Linear(m,k))

inp=torch.randn(n)
writer.add_graph(net, inp)
writer.close()

import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

epoch_list=torch.rand(6)
loss_list=torch.rand(6)
for it in range(len(epoch_list)):
    print(epoch_list[it])
    print(loss_list[it])
    writer.add_scalar("Loss/train", loss_list[it].item(), epoch_list[it].item())

writer.close()

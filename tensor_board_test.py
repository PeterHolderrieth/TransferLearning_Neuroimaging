import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

epoch_list=[1,2,3,4,5,6]
loss_list=[1,2,3,4,5,6]
for it in range(len(epoch_list)):
    writer.add_scalar("Loss/train", loss_list[it], epoch_list[it])

writer.flush()
writer.close()

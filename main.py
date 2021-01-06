from dp_model.model_files.sfcn import SFCN
from dp_model import dp_loss as dpl
from dp_model import dp_utils as dpu
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")  
    print("Running on the GPU")
else:
    DEVICE = torch.device("cpu")
    print("Running on the CPU")


# Example
model = SFCN()
model = torch.nn.DataParallel(model)
fp_ = './pre_trained_models/brain_age/run_20190719_00_epoch_best_mae.p'
model.load_state_dict(torch.load(fp_,map_location=DEVICE))

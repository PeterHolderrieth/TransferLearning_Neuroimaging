#import torch
#from torch.utils.tensorboard import SummaryWriter
#from tensorboard.plugins.hparams import api as hp
'''
writer = SummaryWriter()

epoch_list=torch.tensor([1,2,3,4,5,6])
loss_list=torch.rand(6)
for it in range(len(epoch_list)):
    print(epoch_list[it])
    print(loss_list[it])
    writer.add_scalar("Loss/train", loss_list[it].item(), epoch_list[it].item())

writer.close()
'''

#HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
#HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
#HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

'''
METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )
'''
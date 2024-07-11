import os
import numpy as np
import torch

from config import cfg

#------------prepare enviroment------------
seed = cfg.SEED
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

gpus = cfg.GPU_ID
if len(gpus)==1:
    torch.cuda.set_device(gpus[0])

torch.backends.cudnn.benchmark = True


#------------prepare data loader------------
data_mode = cfg.DATASET
if data_mode == 'SHHA':
    from datasets.SHHA.loading_data import loading_data 
    from datasets.SHHA.setting import cfg_data 
elif data_mode == 'SHHB':
    from datasets.SHHB.loading_data import loading_data 
    from datasets.SHHB.setting import cfg_data 
elif data_mode == 'QNRF':
    from datasets.QNRF.loading_data import loading_data 
    from datasets.QNRF.setting import cfg_data 
elif data_mode == 'WE':
    from datasets.WE.loading_data import loading_data 
    from datasets.WE.setting import cfg_data 


#------------Prepare Tester------------
net = cfg.NET
if net in ['MCNN', 'CSRNet', 'TinyCount']:
    from tester import Tester


#------------Start Testing------------
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    pwd = os.path.split(os.path.realpath(__file__))[0]
    cc_trainer = Tester(loading_data, cfg_data, pwd)
    cc_trainer.forward()


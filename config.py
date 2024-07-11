from easydict import EasyDict as edict
import time

# init
__C = edict()
cfg = __C

#------------------------------TRAIN & TEST------------------
__C.SEED = 3035 # random seed,  for reproduction
__C.DATASET = 'SHHB' # dataset selection: SHHA, SHHB, QNRF, WE


# network config
# net selection - train: TinyCount, MCNN, CSRNet, SANet 
# net selection - test: TinyCount, MCNN, CSRNet 
__C.NET = 'TinyCount'

#------------------------------TRAIN------------------------
__C.RESUME = False # continue training
__C.RESUME_PATH = './exp/..../latest_state.pth' 

__C.GPU_ID = [0] # sigle gpu: [0], [1] ...; multi gpus: [0,1]

# learning rate settings
__C.LR = 1e-4 # learning rate
__C.LR_DECAY = 1 # decay rate
__C.LR_DECAY_START = 1000 # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 20 # decay frequency
__C.MAX_EPOCH = 20

# multi-task learning weights, no use for single model, such as MCNN, VGG, VGG_DECODER, Res50, CSRNet, and so on
__C.LAMBDA_1 = 1e-4 # SANet:0.001

# print 
__C.PRINT_FREQ = 10

now = time.strftime("%m-%d_%H-%M-%S", time.localtime())

__C.EXP_NAME = now \
    + '_' + __C.DATASET \
    + '_' + __C.NET \
    + '_' + str(__C.LR)

__C.EXP_PATH = './exp' # the path of logs, checkpoints, and current codes


#------------------------------VAL------------------------
__C.VAL_DENSE_START = 300
__C.VAL_FREQ = 10 # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ


#------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1 #  must be 1 for training images with the different sizes


#------------------------------Test------------------------
__C.TEST_PATH = './pth/TinyCount_SHHB.pth' 



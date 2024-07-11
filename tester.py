from collections import OrderedDict
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from models.CC import CrowdCounter
from config import cfg
from misc.utils import *


class Tester():
    def __init__(self, dataloader, cfg_data, pwd):

        self.cfg_data = cfg_data

        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.NET
        self.net = CrowdCounter(cfg.GPU_ID,self.net_name).cuda()

        self.timer = {'iter time' : Timer(),'train time' : Timer(),'val time' : Timer()} 

        _, self.val_loader, self.restore_transform = dataloader()

        latest_state = torch.load(cfg.TEST_PATH)
        self.net.load_state_dict(latest_state)

        if cfg.DATASET == 'SHHA':
            self.selected_dataset = 'ShanghaiTech Part A'
        elif cfg.DATASET == 'SHHB':
            self.selected_dataset = 'ShanghaiTech Part B'
        elif cfg.DATASET == 'QNRF':
            self.selected_dataset = 'UCF-QNRF'
        elif cfg.DATASET == 'WE':
            self.selected_dataset = 'WorldExpo\'10'
        


    def forward(self):
        print(f'Selected dataset: {self.selected_dataset}, model: {self.net_name} ...')
        self.timer['val time'].tic()
        if self.data_mode in ['SHHA', 'SHHB', 'QNRF', 'UCF50']:
            self.validate_V1()
        elif self.data_mode == 'WE':
            self.validate_V2()
        self.timer['val time'].toc(average=False)
        print( 'test time: {:.2f}s'.format(self.timer['val time'].diff) )           


    def validate_V1(self):# validate_V1 for SHHA, SHHB, UCF-QNRF

        self.net.eval()
        
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()

                pred_map = self.net.forward(img,gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                
                    pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                    losses.update(self.net.loss.item())
                    maes.update(abs(gt_count-pred_cnt))
                    mses.update((gt_count-pred_cnt)*(gt_count-pred_cnt))
            
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        print_test_summary([mae, mse, loss])


    def validate_V2(self):# validate_V2 for WE

        self.net.eval()

        losses = AverageCategoryMeter(5)
        maes = AverageCategoryMeter(5)

        roi_mask = []
        from datasets.WE.setting import cfg_data 
        from scipy import io as sio
        for val_folder in cfg_data.VAL_FOLDER:

            roi_mask.append(sio.loadmat(os.path.join(cfg_data.DATA_PATH,'test',val_folder + '_roi.mat'))['BW'])
        
        for i_sub,i_loader in enumerate(self.val_loader,0):

            mask = roi_mask[i_sub]
            for vi, data in enumerate(i_loader, 0):
                img, gt_map = data

                with torch.no_grad():
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()

                    pred_map = self.net.forward(img,gt_map)

                    pred_map = pred_map.data.cpu().numpy()
                    gt_map = gt_map.data.cpu().numpy()

                    for i_img in range(pred_map.shape[0]):
                    
                        pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                        gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                        losses.update(self.net.loss.item(),i_sub)
                        maes.update(abs(gt_count-pred_cnt),i_sub)
            
        mae = np.average(maes.avg)
        loss = np.average(losses.avg)

        print_test_summary([mae, 0, loss])


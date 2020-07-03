import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from dataset_loader import MyData, MyTestData
from functions import imsave
import argparse
from train import Trainer
from utils.evaluateFM import get_FM
from model_depth import DepthNet
from model_baseline import BaselineNet
from model_ladder import LadderNet
import time

from yacs.config import CfgNode as CN
import torchvision

import os

configurations = {
    1: dict(
        max_iteration=1000000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        spshot=20000,
        nclass=2,
        sshow=10,
    ),
    'stage2_cfg': dict(
        NUM_BRANCHES = 2,
        NUM_CHANNELS = [32, 64],
        NUM_BLOCKS = [4, 4],
    ),
    'stage3_cfg': dict(
        NUM_BRANCHES = 3,
        NUM_CHANNELS=[32, 64, 128],
        NUM_BLOCKS=[4, 4, 4],
    ),
    'stage4_cfg': dict(
        NUM_MODULES = 1,
        NUM_BRANCHES = 4,
        NUM_BLOCKS = [4, 4, 4, 4],
        NUM_CHANNELS = [32, 64, 128, 256],
    )
}

parser=argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='test', help='train or test')
parser.add_argument('--param', type=str, default=True, help='path to pre-trained parameters')
parser.add_argument('--train_dataroot', type=str, default='/home/lmk/Files/SOD-RGBD/Training Data/bi-train-data', help='path to train data')
parser.add_argument('--test_dataroot', type=str, default='/home/lmk/Files/SOD-RGBD/Testing Data/DUT-RGBD/test_data', help='path to test data')
parser.add_argument('--snapshot_root', type=str, default='./snapshot', help='path to snapshot')
parser.add_argument('--salmap_root', type=str, default='./sal_map/DUT-RGBD/', help='path to saliency map')
parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys())
args = parser.parse_args()
cfg = configurations
cuda = torch.cuda.is_available

"""""""""""dataset loader"""""""""

train_dataRoot = args.train_dataroot
test_dataRoot = args.test_dataroot

if not os.path.exists(args.snapshot_root):
    os.mkdir(args.snapshot_root)
if not os.path.exists(args.salmap_root):
    os.mkdir(args.salmap_root)

if args.phase == 'train':
    SnapRoot = args.snapshot_root           # checkpoint
    train_loader = torch.utils.data.DataLoader(MyData(train_dataRoot, transform=True),
                                               batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
else:
    MapRoot = args.salmap_root
    test_loader = torch.utils.data.DataLoader(MyTestData(test_dataRoot, transform=True),
                                   batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
print ('data already')

"""""""""""train_data/test_data through nets"""""""""

start_epoch = 0
start_iteration = 0

model_depth = DepthNet()
model_baseline = BaselineNet()
model_ladder = LadderNet(cfg)

# print(model_rgb)

# 批量test code ： 在train的时候，去掉外层for循环即可
f = open("DUT-RGBDresult.txt", mode='a+')   #去掉
for ckpt in range(90,91):   #去掉
    if args.param is True:
        ckpt = str(ckpt)
        model_depth.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'depth_snapshot_iter_' + ckpt + '0000.pth')))
        model_baseline.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'baseline_snapshot_iter_'+ckpt+'0000.pth')))
        model_ladder.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'ladder_snapshot_iter_'+ckpt+'0000.pth')))
        # model_rgb.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'snapshot_iter_800000.pth')))
    else:

        model_depth.init_weights()
        vgg19_bn = torchvision.models.vgg19_bn(pretrained=True)
        model_baseline.copy_params_from_vgg19_bn(vgg19_bn)
        model_ladder.init_weights()


    if cuda:
       model_depth = model_depth.cuda()
       model_baseline = model_baseline.cuda()
       model_ladder = model_ladder.cuda()


    if args.phase == 'train':

        # Trainer: class, defined in trainer.py
        optimizer_depth = optim.SGD(model_depth.parameters(), lr=cfg[1]['lr'], momentum=cfg[1]['momentum'], weight_decay=cfg[1]['weight_decay'])
        optimizer_baseline = optim.SGD(model_baseline.parameters(), lr=cfg[1]['lr'], momentum=cfg[1]['momentum'], weight_decay=cfg[1]['weight_decay'])
        optimizer_ladder = optim.SGD(model_ladder.parameters(), lr=cfg[1]['lr'], momentum=cfg[1]['momentum'], weight_decay=cfg[1]['weight_decay'])

        training = Trainer(
            cuda=cuda,
            model_depth=model_depth,
            model_baseline=model_baseline,
            model_ladder=model_ladder,
            optimizer_depth=optimizer_depth,
            optimizer_baseline=optimizer_baseline,
            optimizer_ladder=optimizer_ladder,
            train_loader=train_loader,
            max_iter=cfg[1]['max_iteration'],
            snapshot=cfg[1]['spshot'],
            outpath=args.snapshot_root,
            sshow=cfg[1]['sshow']
        )
        training.epoch = start_epoch
        training.iteration = start_iteration
        training.train()
    else:
        res = []
        for id, (data, depth, img_name, img_size) in enumerate(test_loader):
            # print('testing bach %d' % id)

            inputs = Variable(data).cuda()
            depth = Variable(depth).cuda()
            n, c, h, w = inputs.size()
            # depth = torch.unsqueeze(depth, 1)
            depth = depth.view(n, 1, h, w).repeat(1, c, 1, 1)
            torch.cuda.synchronize()
            start = time.time()
            h2, h3, h4, h5 = model_baseline(inputs)
            d2, d3, d4 = model_depth(depth)
            
            predict_stage2_mask, predict_stage3_mask, predict_stage4_mask = model_ladder(h2, h3, h4, h5, d2, d3, d4)

            torch.cuda.synchronize()
            end = time.time()


            res.append(end - start)


            outputs_all = F.softmax(predict_stage4_mask, dim=1)
            outputs = outputs_all[0][1]
            outputs = outputs.cpu().data.resize_(h, w)

            imsave(os.path.join(MapRoot,img_name[0] + '.png'), outputs, img_size)
            # imsave(os.path.join(MapRoot,img_name[0][-1] + '.png'), outputs, img_size)
        time_sum = 0
        for i in res:
            time_sum += i
        print("FPS: %f" % (1.0 / (time_sum / len(res))))

        # -------------------------- validation --------------------------- #
        torch.cuda.empty_cache()
        print('the testing process has finished!')
        # print("\nevaluating mae....")
        F_measure, mae = get_FM(salpath=MapRoot+'/', gtpath=test_dataRoot+'/test_masks/')
        print('F_measure:', F_measure)
        print('MAE:', mae)
        F_measure = str(F_measure)
        mae = str(mae)
        f.write(ckpt + '0000.pth : \t' + F_measure + '\t' + mae + '\n')  # 去掉
f.close()  # 去掉

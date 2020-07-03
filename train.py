import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch

running_loss_final = 0# 这是什么


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0] # 262144 #input = 2*256*256*2
    input = input.view(-1, c)
    mask = target >= 0
    target = target[mask]
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


class Trainer(object):

    def __init__(self, cuda, model_depth, model_baseline, model_ladder,  optimizer_depth, optimizer_baseline, optimizer_ladder,
                 train_loader, max_iter, snapshot, outpath, sshow, size_average=False):
        self.cuda = cuda
        self.model_depth = model_depth
        self.model_baseline = model_baseline
        self.model_ladder = model_ladder
        self.optim_depth = optimizer_depth
        self.optim_baseline = optimizer_baseline
        self.optim_ladder = optimizer_ladder
        self.train_loader = train_loader
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.snapshot = snapshot
        self.outpath = outpath
        self.sshow = sshow
        self.size_average = size_average

    def train_epoch(self):

        for batch_idx, (img, mask, depth) in enumerate(self.train_loader):

            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration
            if self.iteration >= self.max_iter:
                break
            if self.cuda:
                img, mask, depth = img.cuda(), mask.cuda(), depth.cuda()
                img, mask, depth = Variable(img), Variable(mask), Variable(depth)
            n, c, h, w = img.size()  # batch_size, channels, height, weight

            self.optim_depth.zero_grad()
            self.optim_baseline.zero_grad()
            self.optim_ladder.zero_grad()

            global running_loss_final 

            # depth = torch.unsqueeze(depth, 1)
            depth = depth.view(n, 1, h, w).repeat(1, c, 1, 1)


            d2, d3, d4 = self.model_depth(depth)
            h2, h3, h4, h5 = self.model_baseline(img)

            predict_stage2_mask, predict_stage3_mask, predict_stage4_mask = self.model_ladder(h2, h3, h4, h5, d2, d3, d4)  # RGBNet's output 待修改

            loss_stage4_mask = cross_entropy2d(predict_stage4_mask, mask, size_average=False)
            loss_all = loss_stage4_mask
            running_loss_final += loss_all.item()

            if iteration % self.sshow == (self.sshow - 1):
                print('\n [%3d, %6d,   RGB-D Net loss: %.3f]' % (
                self.epoch + 1, iteration + 1, running_loss_final / (n * self.sshow)))

                running_loss_final = 0.0

            if iteration <= 200000:
                if iteration % self.snapshot == (self.snapshot - 1):
                    savename_depth = ('%s/depth_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_depth.state_dict(), savename_depth)
                    print('save: (snapshot: %d)' % (iteration + 1))

                    savename_baseline = ('%s/baseline_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_baseline.state_dict(), savename_baseline)
                    print('save: (snapshot: %d)' % (iteration + 1))

                    savename_ladder = ('%s/ladder_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_ladder.state_dict(), savename_ladder)
                    print('save: (snapshot: %d)' % (iteration + 1))
            else:

                if iteration % 10000 == (10000 - 1):
                    savename_depth = ('%s/depth_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_depth.state_dict(), savename_depth)
                    print('save: (snapshot: %d)' % (iteration + 1))

                    savename_baseline = ('%s/baseline_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_baseline.state_dict(), savename_baseline)
                    print('save: (snapshot: %d)' % (iteration + 1))

                    savename_ladder = ('%s/ladder_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_ladder.state_dict(), savename_ladder)
                    print('save: (snapshot: %d)' % (iteration + 1))

            if (iteration + 1) == self.max_iter:
                savename_depth = ('%s/depth_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                torch.save(self.model_depth.state_dict(), savename_depth)
                print('save: (snapshot: %d)' % (iteration + 1))

                savename_baseline = ('%s/baseline_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                torch.save(self.model_baseline.state_dict(), savename_baseline)
                print('save: (snapshot: %d)' % (iteration + 1))

                savename_ladder = ('%s/ladder_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                torch.save(self.model_ladder.state_dict(), savename_ladder)
                print('save: (snapshot: %d)' % (iteration + 1))

            loss_all.backward()
            self.optim_depth.step()
            self.optim_baseline.step()
            self.optim_ladder.step()

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))

        for epoch in range(max_epoch):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break

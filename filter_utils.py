import os
import shutil
import torch
from torchvision.models import AlexNet
from torch.optim.lr_scheduler import CosineAnnealingLR,ExponentialLR
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import _LRScheduler



def fiter_fail_dir(path='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn'):
    dir_list = os.listdir(path)

    for dir_name in dir_list:
        dir_path = os.path.join(path,dir_name,'checkpoints')

        file_len = len(os.listdir(dir_path))


        if file_len<50:
            shutil.rmtree(os.path.join(path, dir_name))

        # if not os.path.exists(dir_path):
            # break



class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def lr_test():

    model = AlexNet(num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # scheduler = CosineAnnealingLR(optimizer, T_max=200,eta_min=0.0001)
    warmup_scheduler = WarmUpLR(optimizer, 2000)
    scheduler = ExponentialLR(optimizer,0.97)
    plt.figure()
    x = list(range(200))
    y = []
    for epoch in range(1, 201):
        if epoch<20:
            warmup_scheduler.step()
        else:
            scheduler.step()


        optimizer.zero_grad()
        optimizer.step()
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))

        y.append(scheduler.get_lr()[0])

    # 画出lr的变化
    plt.plot(x, y)
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.title("learning rate's curve changes as epoch goes on!")
    plt.show()


def cur_lr(epoch):
    return 0.001 * 0.3 ** (epoch // 500)

if __name__ == '__main__':
    lr_test()

    # plt.figure()
    # x = list(range(1000))
    # y = []
    # for epoch in range(0, 1000):
    #
    #     y.append(cur_lr(epoch))
    #
    # plt.plot(x, y)
    # plt.xlabel("epoch")
    # plt.ylabel("lr")
    # plt.title("learning rate's curve changes as epoch goes on!")
    # plt.show()



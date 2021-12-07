import argparse
import csv
from easydict import EasyDict as edict
import yaml
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import lib.utils.utils as utils
from lib.dataset import get_dataset
from lib.core import function
import lib.config.lp_name as alphabets
from lib.utils.utils import model_info
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler



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

def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str,default="lib/config/lp_config.yaml")
    parser.add_argument('--name', help='net type', type=str,default="crnn")
    parser.add_argument('--lr_declay', help='net type', type=str,default="step")

    args = parser.parse_args()



    with open(args.cfg, 'r') as f:
        # config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.load(f)
        config_ = config
        config = edict(config)
        

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    config.MODEL.NAME = args.name
    config.LRDECLAY = args.lr_declay

    config_['DATASET']['ALPHABETS'] = alphabets.alphabet
    config_['MODEL']['NUM_CLASSES'] = len(config.DATASET.ALPHABETS)
    config_['MODEL']['NAME'] = args.name
    config_['LRDECLAY'] = args.lr_declay

    return config,config_

def main():

    # load config
    config,config_ = parse_arg()
    # create output folder and save config
    output_dict = utils.create_log_folder(config,config_,phase='train')

    # cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # writer dict
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # construct face related neural networks
    if config.MODEL.NAME == 'crnn':
        import lib.models.crnn as crnn
    else:
        import lib.models.cnn_att_ctc as crnn
    model = crnn.get_crnn(config)

    # get device
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.GPUID))
    else:
        device = torch.device("cpu:0")

    model = model.to(device)

    # define loss function
    criterion = torch.nn.CTCLoss()

    last_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = utils.get_optimizer(config, model)


    # finetine train
    if config.TRAIN.FINETUNE.IS_FINETUNE:
        model_state_file = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        from collections import OrderedDict
        model_dict = OrderedDict()
        for k, v in checkpoint.items():
            if 'cnn' in k:
                model_dict[k[4:]] = v
        model.cnn.load_state_dict(model_dict)
        if config.TRAIN.FINETUNE.FREEZE:
            for p in model.cnn.parameters():
                p.requires_grad = False
    # resum train
    elif config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            model.load_state_dict(checkpoint)

    # model params statistics
    model_info(model)

    # datasets and dataloder
    train_dataset = get_dataset(config)(config, is_train=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_dataset = get_dataset(config)(config, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    # warm_up setting
    warm_up_epoch = 1
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, warm_up_epoch * iter_per_epoch)

    # learning_rate_declay 
    if config.LRDECLAY == "step":
        if isinstance(config.TRAIN.LR_STEP, list):
            train_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, config.TRAIN.LR_STEP,
                config.TRAIN.LR_FACTOR, last_epoch-1
            )
        else:
            train_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, config.TRAIN.LR_STEP,
                config.TRAIN.LR_FACTOR, last_epoch - 1
            )

    else:
        train_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.97)


    # save min_acc set
    best_acc = 0.5
    
    # label convert num
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

    # start training
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):

        # train
        function.train(config, train_loader, train_dataset, converter, model, criterion, optimizer, device, epoch,writer_dict=writer_dict,warmup_scheduler=warmup_scheduler,train_scheduler=train_scheduler,warm_up_epoch=warm_up_epoch)
        # validate
        acc = function.validate(config, val_loader, val_dataset, converter, model, criterion, device, epoch, writer_dict, output_dict)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        print("is best:", is_best)
        print("best acc is:", best_acc)
        # save checkpoint
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "optimizer": optimizer.state_dict(),
                "train_scheduler": train_scheduler.state_dict(),
                "best_acc": best_acc,
            },  os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
        )

    writer_dict['writer'].close()

if __name__ == '__main__':

    main()


# 12.16 mobilenetv3:(0.35)   SE_block:False transform_layer_num:5->3
# 03.09 mobilenetv3:(larger) SE_block:True  transform_layer_num:5
# 增加Transforme_Encoder模块数量（发现增加后网络不容易收敛、容易梯度弥散）
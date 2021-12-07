import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.att_arcloss as crnn
import lib.config.lp_name as alphabets
import yaml
from easydict import EasyDict as edict
import argparse
import os
# '/home/data-set/internal/nunova_platenumberv2.0/testg/',


flase_dict = {}

def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str,
                        default='lib/config/lp_config.yaml')

    parser.add_argument('--image_dir', type=str,
                        # default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/imgge',
                        # default='/home/hyli/project/OCR/LPRNet_Pytorch/data/ZSDX_blue_test_no_yellow/',
                        # default='/home/data-set/internal/lp_video_extract',
                        # default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/images',
                        default='/home/data-set/internal/nunova_platenumberv2.0/testg/',



                        help='the path to your image')
    parser.add_argument('--checkpoint', type=str,
                        # default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2020-11-25-14-56/checkpoints/checkpoint_67_acc_0.9112.pth',
                        # default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2020-11-25-19-17/checkpoints/checkpoint_93_acc_0.9084.pth',
                        # default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2020-11-23-09-56/checkpoints/checkpoint_31_acc_0.9309.pth',
                        # default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2020-12-03-15-17/checkpoints/checkpoint_180_acc_0.7849.pth',
                        default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2020-12-07-20-18/checkpoints/checkpoint_199_acc_0.4529.pth',
                        # default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2020-12-06-10-26/checkpoints/checkpoint_199_acc_0.7673.pth',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args


def judge_len(pred,label):
    if len(pred)!=len(label):
        return 1
    else:
        return 0

def judge_first(pred,label):
    if pred[0]!=label[0]:
        return 1
    else:
        return 0


def judge_correct(pred,label):
    global flase_dict
    if pred==label:
        return 1
    else:
        if pred in flase_dict.keys():
            flase_dict[pred] += 1
        else:
            flase_dict[pred] = 1
        # if len(pred) == len(label):
        #     for i in range(len(pred)):
        #         if pred[i] != label[i]:
        #
        #             if pred[i]in flase_dict.keys():
        #
        #                 flase_dict[pred[i]] +=1
        #             else:
        #                 flase_dict[pred[i]] = 1

        print("pred:{}-------label:{}".format(pred, label))
        return 0




def recognition(config, img, model, converter, device,image_name=None):
    img_h, img_w,_ = img.shape

    # img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.W / img_w, fy=config.MODEL.IMAGE_SIZE.H / img_h,
    #                  interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, (config.MODEL.IMAGE_SIZE.W, config.MODEL.IMAGE_SIZE.H),  interpolation=cv2.INTER_CUBIC)

    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W, 3))


    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])

    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img).permute(1, 0, 2)
    # print(preds.shape)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    label = image_name.split(".")[0]

    # print("pred:{}-------label:{}".format(sim_pred, label))


    return judge_correct(sim_pred,label),judge_len(sim_pred,label),judge_first(sim_pred,label)


if __name__ == '__main__':

    config, args = parse_arg()
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    model = crnn.get_crnn(config).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    started = time.time()

    lp_sum = len(os.listdir(args.image_dir))
    right_sum = 0
    first_wrong = 0
    len_wrong = 0



    for image_name in os.listdir(args.image_dir):
        img_path = os.path.join(args.image_dir,image_name)
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
        flag,len_flag,first_flag = recognition(config, img, model, converter, device,image_name)
        right_sum += flag
        len_wrong += len_flag

        first_wrong += first_flag



    finished = time.time()

    acc = right_sum/lp_sum
    len_wrong_rate = len_wrong/(lp_sum-right_sum)
    first_wrong_rate = first_wrong/(lp_sum-right_sum)
    string_rec_wrong = (lp_sum-right_sum-len_wrong-first_wrong)/(lp_sum-right_sum)
    print('elapsed time: {0}'.format(finished - started))
    print("lp_sum:{}--right_sum:{}---wrong_sum:{}---len_wrong:{}---first_wrong:{}".format(lp_sum, right_sum,lp_sum-right_sum,len_wrong,first_wrong))
    print("acc---:{:.4}---len_wrong_rate:{:.4}---first_wrong_rate:{:.4}---string_rec_wrong:{:.4}".format(acc,len_wrong_rate,first_wrong_rate,string_rec_wrong))
    print(flase_dict)


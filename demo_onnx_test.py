import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
# import lib.models.cnn_att_ctc as crnn
import lib.config.lp_name as alphabets
import yaml
from easydict import EasyDict as edict
import argparse
import os
import onnxruntime as ort




flase_dict = {}

def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str,
                        default='lib/config/lp_config.yaml')
    parser.add_argument('--image_dir', type=str,
                        # default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/imgge',
                        # default='/home/hyli/project/OCR/LPRNet_Pytorch/data/ZSDX_blue_test_no_yellow/',
                        # default='/home/data-set/internal/lp_video_extract',
                        # default='/home/data-set/internal/nunova_platenumberv2.0/test/',
                        default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/test',
                        # default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/test',


                        help='the path to your image')
    parser.add_argument('--checkpoint', type=str,
                        # default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2020-11-25-14-56/checkpoints/checkpoint_67_acc_0.9112.pth',
                        # default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2020-11-25-19-17/checkpoints/checkpoint_93_acc_0.9084.pth',
                        # default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2020-11-23-09-56/checkpoints/checkpoint_31_acc_0.9309.pth',
                        # default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2020-12-03-15-17/checkpoints/checkpoint_180_acc_0.7849.pth',
                        default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2020-12-04-05-05/checkpoints/checkpoint_117_acc_0.8057.pth',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args


def judge_len(pred,label,scores,len_file=None):
    if len(pred)!=len(label):
        len_file.write("pred:{}-------label:{}\n".format(pred, label))
        len_file.write(str(scores))
        len_file.write('\n')
        return 1
    else:
        return 0

def judge_first(pred,label,scores,first_file=None):
    if pred[0]!=label[0]:
        first_file.write("pred:{}-------label:{}\n".format(pred, label))
        first_file.write(str(scores))
        first_file.write('\n')
        return 1
    else:
        return 0


def judge_correct(pred,label,scores,other_file=None):
    global flase_dict
    if pred==label:
        return 1
    else:
        if len(pred) == len(label):
            for i in range(len(pred)):
                if pred[i] != label[i]:

                    if pred[i] in flase_dict.keys():

                        flase_dict[pred[i]] +=1
                    else:
                        flase_dict[pred[i]] = 1
        other_file.write("pred:{}-------label:{}\n".format(pred, label))
        other_file.write(str(scores))
        other_file.write('\n')
        # print("==========")
        # print("pred:{}-------label:{}".format(pred, label))
        # print(scores)
        return 0


def cal_score(pres,idx):

    pres = pres.reshape(-1,84)
    idx = idx.reshape(-1)


    sclore = 0

    for i in range(40):

        sclore += pres[i][idx[i]]

    return sclore/40




def recognition(config, img, model, converter, device,image_name=None,len_dict=None,len_file=None, first_file=None, other_file=None):


    # data pre_process
    img_h, img_w,_ = img.shape
    img = cv2.copyMakeBorder(img, 2, 2, 6, 6, cv2.BORDER_REPLICATE)
    img = cv2.resize(img, (config.MODEL.IMAGE_SIZE.W, config.MODEL.IMAGE_SIZE.H),  interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W, 3))
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)
    img = img.to(device)
    input = img.view(1, *img.size())

    # model infer
    preds = model.run(None, {'input': input.cpu().numpy().astype(np.float32)})[0]


    preds = torch.nn.functional.softmax(torch.Tensor(preds), dim=-1).numpy() 

    # print(preds)

    # post_process
    preds_ = preds.argmax(2)
    preds_ = preds_.transpose(1, 0).reshape(-1)
    sim_pred,scores = converter.numpy_decode((preds_.data),preds,len_dict=len_dict)
    label = image_name.split(".")[0]

    print(sim_pred,label)
    print(scores)

    # if scores < 0.95:
    # print("pred:{}-------label:{}----scores:{}".format(sim_pred, label,scores))


    return judge_correct(sim_pred,label,scores,other_file=other_file),judge_len(sim_pred,label,scores,len_file=len_file),judge_first(sim_pred,label,scores,first_file=first_file)


def infer_recognition(config, img, model, converter, device,len_dict=None):


    img_h, img_w,_ = img.shape

    # img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.W / img_w, fy=config.MODEL.IMAGE_SIZE.H / img_h,
    #                  interpolation=cv2.INTER_CUBIC)

    img = cv2.copyMakeBorder(img, 2, 2, 6, 6, cv2.BORDER_REPLICATE)
    img = cv2.resize(img, (config.MODEL.IMAGE_SIZE.W, config.MODEL.IMAGE_SIZE.H),  interpolation=cv2.INTER_CUBIC)

    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W, 3))


    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])

    img = torch.from_numpy(img)

    img = img.to(device)
    input = img.view(1, *img.size())
    # print(input.shape)
    # model.eval()
    # preds = model(img).permute(1, 0, 2)
    preds = model.run(None, {'input': input.cpu().numpy().astype(np.float32)})[0]


    
    # print(preds.size())
    preds_ = preds.argmax(2)
    preds_ = preds_.transpose(1, 0).reshape(-1)


    # score = cal_score(preds,preds_)
    # print(score)


    sim_pred,scores = converter.numpy_decode((preds_.data),preds,len_dict=len_dict)
    print(sim_pred)



    return  sim_pred






def net_lp_data_mk(config,lp_dir="/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/lib/dataset",save_path='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/lp_database'):
    lp_ocr = Lp_Infer(config=config)
    num = 0
    data_root = os.listdir(lp_dir)
    for data_first in data_root:

        data_second = os.path.join(lp_dir,data_first)
        if not os.path.isdir(data_second):
            continue

        if int(data_first)<20201202 :
            continue

        if os.path.isdir(data_second):
            for data_tir in os.listdir(data_second):
                lp_path = os.path.join(data_second,data_tir)
                for lp_name in os.listdir(lp_path):
                    lp_root = os.path.join(lp_path,lp_name)
                    lp = lp_ocr.infer(lp_root)
                    lp_ocr.save_img_(lp_root,lp,save_path)
                    num += 1
                    print(num)

    print(lp_ocr.num)


class Lp_Infer:
    def __init__(self,config):
        self.net = ort.InferenceSession('./.onnx')
        self.config = config
        self.converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
        self.device = torch.device('cuda:1')
        self.num = 0

    def infer(self,img_path):
        img = cv2.imread(img_path)
        infer_lp = infer_recognition(self.config,img,self.net, self.converter, self.device, len_dict=None)
        print(infer_lp)

        return infer_lp

    def save_img_(self,img_path,lp_name,save_path):

        img_ = cv2.imread(img_path)

        img_path_ = "{}/{}.jpg".format(save_path,lp_name)
        if not os.path.exists(img_path_):
            cv2.imwrite(img_path_, img_)
            self.num += 1
            print("save_done")


def false_type_classify():
    len_file = open('./test_analysis/len_wrong.txt','w')
    first_file = open('./test_analysis/first_str_wrong.txt','w')
    other_file = open('./test_analysis/other_str_wrong.txt','w')
    return len_file,first_file,other_file





if __name__ == '__main__':

    # parse config
    config, args = parse_arg()
    # set train device
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    # # device = torch.device('cpu')

    # create flase type file
    len_file, first_file, other_file = false_type_classify()
    # start_time
    started = time.time()



    # acc_count_ele
    lp_sum = len(os.listdir(args.image_dir))
    right_sum = 0
    first_wrong = 0
    len_wrong = 0
    ort_session = ort.InferenceSession('./crnn.onnx')
    len_dict = {}


    for image_name in os.listdir(args.image_dir).sort():
        img_path = os.path.join(args.image_dir,image_name)
        img = cv2.imread(img_path)
        converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
        flag,len_flag,first_flag = recognition(config, img, ort_session, converter, device,image_name,len_dict=len_dict,len_file=len_file, first_file=first_file, other_file=other_file)
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
    print(sorted(len_dict.items(),key=lambda kv:(kv[1],kv[0]),reverse=True))
    print(sorted(flase_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))


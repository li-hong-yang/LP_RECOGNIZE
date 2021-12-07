import numpy as np
import time
import cv2
import torch
import torch.nn as nn
import lib.config.lp_name as alphabets
import yaml
from easydict import EasyDict as edict
import argparse
import os
import io
from PIL import Image
import torch.nn.functional as F

class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1
    def decode(self,t,preds):
        preds = preds.reshape(41,84)
        char_list = []
        for i in range(len(t)):
            if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                char_list.append(self.alphabet[t[i] - 1])
        lp = ''.join(char_list)
        return lp

   

class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):

        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        # print(conv.size())
        # assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # b *512 * width
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = F.log_softmax(self.rnn(conv), dim=2)
        return output

def get_crnn(config):
    model = CRNN(config.MODEL.IMAGE_SIZE.H, 3, config.MODEL.NUM_CLASSES + 1, config.MODEL.NUM_HIDDEN)
    return model

def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str,
                        default='lib/config/lp_config.yaml')
    parser.add_argument('--image_dir', type=str,
                        default="/home/hyli/project/OCR/OCR_datasets/testg/",
                        help='the path to your image')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2021-03-09-18-35/checkpoints/checkpoint_95_acc_0.8467.pth',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args



class LpInfer:
    def __init__(self,config,args,device):
        self.device = device
        self.config = config
        self.args = args
        self.model = get_crnn(config).to(device)
        self.converter = strLabelConverter(config.DATASET.ALPHABETS)
        self.load_weights()

    def infer_recognition(self,img):
        img = cv2.copyMakeBorder(img, 2, 2, 6, 6, cv2.BORDER_REPLICATE)
        img = cv2.resize(img, (config.MODEL.IMAGE_SIZE.W, config.MODEL.IMAGE_SIZE.H),  interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W, 3))
        img = img.astype(np.float32)
        img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
        img = img.transpose([2, 0, 1])
        img = torch.from_numpy(img)
        img = img.view(1, *img.size())
        preds = self.model(img.to(device))
        preds_ = preds.argmax(2)
        preds_ = preds_.transpose(1, 0).reshape(-1)
        sim_pred = self.converter.decode((preds_.data),preds)
        return  sim_pred

    def infer(self,img_path):
        # img = Image.open(io.BytesIO(img_path))
        img = cv2.imread("/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/1.jpg")
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        infer_lp = self.infer_recognition(img)
        return infer_lp

    def load_weights(self):
        checkpoint = torch.load(self.args.checkpoint)
        if 'state_dict' in checkpoint.keys():
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print("load weights succeed")


# from flask import Flask, jsonify, request
# # flask web
# app = Flask(__name__)
# app.config['JSON_AS_ASCII'] = False

device = torch.device('cuda:2')



config, args = parse_arg()
model =  LpInfer(config,args,device)

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         file = request.files['file']
#         img_bytes = file.read()
#         lp = model.infer(img_bytes)
#         return jsonify({'class_id': lp})


if __name__ == '__main__':
    print(model.infer("img"))



   
      





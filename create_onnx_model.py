import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
# import lib.models.mobilenetv3_crnn as crnn
# import lib.config.alphabets as
# alphabets
# import lib.models.cnn_att_ctc as crnn
# import lib.models.src_net as crnn
import lib.models.crnn as crnn
import lib.config.lp_name as alphabets
import yaml
from easydict import EasyDict as edict
import argparse
import os


# '/home/data-set/internal/nunova_platenumberv2.0/testg/',

def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str,
                        default='lib/config/lp_config.yaml')
    parser.add_argument('--image_dir', type=str,
                        default='/home/data-set/internal/nunova_platenumberv2.0/testg/',
                        # default='/home/hyli/project/OCR/LPRNet_Pytorch/data/ZSDX_blue_test_no_yellow/',
                        # default='/home/data-set/public/test_no_yellow/',



                        help='the path to your image')
    parser.add_argument('--checkpoint', type=str,
                        # default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2020-11-25-14-56/checkpoints/checkpoint_67_acc_0.9112.pth',
                        # default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2020-11-25-19-17/checkpoints/checkpoint_93_acc_0.9084.pth',
                        # default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2020-12-01-19-07/checkpoints/checkpoint_99_acc_0.9004.pth',
                        # default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2020-11-23-09-56/checkpoints/checkpoint_31_acc_0.9309.pth',
                        # default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2020-12-04-05-05/checkpoints/checkpoint_117_acc_0.8057.pth',
                        # default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2020-12-08-21-59/checkpoints/checkpoint_187_acc_0.8202.pth',
                        default='/home/hyli/project/OCR/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2021-03-09-18-35/checkpoints/checkpoint_95_acc_0.8467.pth',# CRNN
                        help='the path to your checkpoints')


    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

import onnxruntime as ort



def onnx_runtime_test():
    ys = torch.randn(size=(1, 3, 32, 160))
    started = time.time()
    ort_session = ort.InferenceSession('crnn.onnx')
    # ort_session = ort.InferenceSession('./onnx_model/crnn_att_3_channels.onnx')
    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))
    outputs = ort_session.run(None, {'input': ys.numpy().astype(np.float32)})
    # outputs = ort_session.run(None, {'input': ys.numpy().astype(np.float32)})
    # print(outputs[0][0,0,0:10])
    # print(outputs[0].shape)
    # print(outputs[0].shape)
    finished_module = time.time()
    print('elapsed time: {0}'.format(finished_module-finished))

def onnx_runtime_att():
    ys = torch.randn(size=(1, 3, 32, 160))
    started = time.time()
    ort_session = ort.InferenceSession('srn.onnx')
    # ort_session = ort.InferenceSession('cr_b1_T.onnx')
    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))
    # outputs = ort_session.run(None, {'input_hwc:0': ys.numpy().astype(np.float32)})
    outputs = ort_session.run(None, {'input': ys.numpy().astype(np.float32)})
    # print(outputs[0][0,0,0:10])
    print(outputs[0].shape)
    # print(outputs[0].shape)
    finished_module = time.time()
    print('elapsed time: {0}'.format(finished_module-finished))

def pytorch_runtime_test(xs):
    config, args = parse_arg()

    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    model = crnn.get_crnn(config).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if 'state_dict' in checkpoint.keys():
        started = time.time()
        model.load_state_dict(checkpoint['state_dict'])
        finished = time.time()
        print('elapsed time: {0}'.format(finished - started))
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    started = time.time()


    output = model(xs)
    print(output[0,0,0:10])
    print(output.shape)
    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))



def create_onnx():
    config, args = parse_arg()
    model = crnn.get_crnn(config)
    checkpoint = torch.load(args.checkpoint)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
        print('loading pretrained model from {0}'.format(args.checkpoint))
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    dummy_input1 = torch.randn(1, 3, 32, 160)
    # torch.onnx.export(model,dummy_input1,'crnn.onnx',verbose=True,input_names=['input'], output_names=['output'])

    torch.onnx.export(model,  # model being run
                      dummy_input1,  # model input (or a tuple for multiple inputs)
                      'crnn.onnx',  # where to save the model (can be a file or file-like object)
                      verbose=True,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})


import torch.nn as nn
import torch.nn.functional as F


class test_module(nn.Module):
    def __init__(self):
        super(test_module, self).__init__()

    def forward(self, x):
        N, C, H, W = x.size()
        # return x.view(N,C,-1)

        h = H
        w = W
        return F.avg_pool2d(x, kernel_size=[h, w])


def create_onnx_():
    from torchvision.models.resnet import resnet50

    model = resnet50(pretrained=False)

    model.eval()
    # print(model)


    # checkpoint = torch.load(args.checkpoint)
    # if 'state_dict' in checkpoint.keys():
    #     model.load_state_dict(checkpoint['state_dict'])
    #     print('loading pretrained model from {0}'.format(args.checkpoint))
    # else:
    #     model.load_state_dict(checkpoint)

    dummy_input1 = torch.randn(1, 3, 160, 160)
    torch.onnx.export(model,dummy_input1,'resnet50.onnx',verbose=True,input_names=['input'], output_names=['output'])
    # torch.onnx.export(model,dummy_input1,'crnn.onnx',verbose=True,input_names=['input_hwc:0'], output_names=['output'])
    # torch.onnx.export(model,  # model being run
    #                   dummy_input1,  # model input (or a tuple for multiple inputs)
    #                   'crnn_.onnx',  # where to save the model (can be a file or file-like object)
    #                   verbose=True,
    #                   export_params=True,  # store the trained parameter weights inside the model file
    #                   opset_version=10,  # the ONNX version to export the model to
    #                   do_constant_folding=False,  # whether to execute constant folding for optimization
    #                   input_names=['input'],  # the model's input names
    #                   output_names=['output'],  # the model's output names
    #                   dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
    #                                 'output': {0: 'batch_size'}})


if __name__ == '__main__':
    create_onnx_()
    onnx_runtime_test()




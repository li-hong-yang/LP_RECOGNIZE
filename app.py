import numpy as np
import time
import cv2
import torch
import yaml
from easydict import EasyDict as edict
import argparse
import os
import onnxruntime as ort
import numpy
from flask import Flask, jsonify, request
import io
from PIL import Image


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        length = []
        result = []
        decode_flag = True if type(text[0])==bytes else False

        for item in text:

            if decode_flag:
                item = item.decode('utf-8','strict')
            length.append(len(item))
            for char in item:
                index = self.dict[char]
                result.append(index)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

    def numpy_decode(self,t,preds,len_dict=None):
        # t = preds.argmax(2) shape[40]
        preds = preds.reshape(-1,84)
        score = []
        char_list = []

        for i in range(len(t)):
            if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                # score.append(preds[i][t[i]])
                char_list.append(self.alphabet[t[i] - 1])
                score.append({self.alphabet[t[i] - 1]:preds[i][t[i]]})

        lp = ''.join(char_list)
        return lp,score

def parse_arg():
    alphabet = """京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHJKLMNPQRSTUVWXYZ港学使警澳挂军北南广沈兰成济海民航空"""
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str,
                        default='lib/config/lp_config.yaml')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def infer_recognition(config, img, model, converter, device,len_dict=None):


    img_h, img_w,_ = img.shape
    img = cv2.copyMakeBorder(img, 2, 2, 6, 6, cv2.BORDER_REPLICATE)
    img = cv2.resize(img, (config.MODEL.IMAGE_SIZE.W, config.MODEL.IMAGE_SIZE.H),  interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W, 3))
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])

    img = torch.from_numpy(img)

    img = img.to(device)
    input_ = img.view(1, *img.size())
    preds = model.run(None, {'input': input_.cpu().numpy().astype(np.float32)})[0]

    preds_ = preds.argmax(2)
    preds_ = preds_.transpose(1, 0).reshape(-1)

    sim_pred,scores = converter.numpy_decode((preds_.data),preds,len_dict=len_dict)
    return  sim_pred




class Lp_Infer:
    def __init__(self,config):
        self.net = ort.InferenceSession('crnn.onnx')
        self.config = config
        self.converter = strLabelConverter(config.DATASET.ALPHABETS)
        self.device = torch.device('cuda:1')
        self.num = 0

    def infer(self,img_path):
        img = Image.open(io.BytesIO(img_path))
        img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
        infer_lp = infer_recognition(self.config,img,self.net, self.converter, self.device, len_dict=None)
        return infer_lp

# flask web
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
config, args = parse_arg()
model = Lp_Infer(config=config)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        lp = model.infer(img_bytes)
        return jsonify({'class_id': lp})


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)


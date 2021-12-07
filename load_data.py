
from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os
import csv

import xlwt
import pandas as pd

import albumentations as A

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新','澳','港',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]


lp ={"0":'川',"1":'云',"2":'沪',"3":'贵',"4":'渝',"5":'冀',"6":'晋',"7":'桂',"8":'陕',"9":'青',"10":'藏',
    "11":'苏',"12":'浙',"13":'皖',"14":'闽',"15":'赣',"16":'鲁',"17":'豫',"18":'鄂',"19":'湘',"20":'粤',
     "21":'蒙',"22":"琼","23":'津',"24":'京',"25":'黑',"26":'辽',"27":'甘',"28":'吉',"29":'宁',"30":'新',
     "31":'港','32':'学',"33":'使',"34":'警',"35":'澳',"36":'挂',"37":'军',"38":"北","39":'南',"40":'广',
     "41":'沈',"42":'兰',"43":'成'}
ref_hist = ['/home/hyli/project/datamentations/merge_nunova/lp_histogram/16054920894716.png']

transform = A.Compose(
                            [
                                # A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, num_flare_circles_lower=60,
                                #                 num_flare_circles_upper=100, src_radius=100, src_color=(255, 0, 0),p=1),
                                # A.RandomRain(drop_length=10,p=0.2),
                                # A.RandomFog(p=0.2),
                                # A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=0.2),
                                A.HistogramMatching(ref_hist,blend_ratio=(0.3,0.5) ,p=0.2),
                                # A.GaussNoise(var_limit=(200,300),always_apply=True)
                                A.MotionBlur(blur_limit=10, p=1),
                                A.Rotate(limit=(5,10), interpolation=cv2.INTER_CUBIC,
                                         border_mode=cv2.BORDER_REFLECT_101, p=0.5)
                            ],
    # bbox_params=A.BboxParams(format='pascal_voc', min_area=1, min_visibility=0.1)
                        )

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}




class LPDataLoader(Dataset):
    def __init__(self, img_dir_list, imgSize, lpr_max_len, PreprocFun=None,istrain=False):

        self.img_paths = []
        self.istrain = istrain

        for img_dir in img_dir_list:
            data_list = os.listdir(img_dir)
            for img_name in data_list:
                self.img_paths.append(os.path.join(img_dir,img_name))

        print(len(self.img_paths))

        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        Image = cv2.imread(filename)
        height, width, _ = Image.shape
        # print(height,width)
        if height != self.img_size[1] or width != self.img_size[0]:

            Image = cv2.resize(Image, tuple(self.img_size))

        # if self.istrain:
        #
        #     """"""
        #     Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
        #     Image = transform(image=Image)
        #     Image = cv2.cvtColor(Image['image'], cv2.COLOR_RGB2BGR)
        #     """"""

        Image = self.PreprocFun(Image)


        filename = filename.split("/")[-1].split(".")[0]


        label = list()
        for c in filename:
            label.append(CHARS_DICT[c])




        # if len(label) == 8:
        #     if self.check(label) == False:
        #         print(filename)
        #         assert 0, "Error label ^~^!!!"

        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    def check(self, label):
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("Error label, Please check!")
            return False
        else:
            return True

    def count_differ_province(self,img_dir_list):

        province_dict = {}

        for img_dir in img_dir_list:
            data_list = os.listdir(img_dir)
            for img_name in data_list:

                if not img_name[0] in province_dict.keys():
                    province_dict[img_name[0]] =1
                else:

                    province_dict[img_name[0]] += 1

        filename = "./filename.csv"
        with open(filename,"w") as csv_file:
            [csv_file.write("{},{}\n".format(key,value)) for key,value in province_dict.items()]



        columns_map = {}

        province_dict = sorted(province_dict.items(),key=lambda x:x[1],reverse=True)
        print(province_dict)


def convert_label(img_dir,save_txt_dir):

    with open(save_txt_dir,"w") as f:
        for img_name in os.listdir(img_dir):
            name_convert = img_name.split(".")[0]

            img_name_ = ""

            for name in name_convert:
                img_name_+=" {}".format(CHARS_DICT[name])
            # print(img_name_)

            f.write(img_name+img_name_+"\n")

if __name__ == '__main__':
    # import tensorflow
    # print(tensorflow.__version__)
    # dirs = ["/home/hyli/project/OCR/LPRNet_Pytorch/data/CCPD_blue", "/home/hyli/project/OCR/LPRNet_Pytorch/data/HC_blue"]
    #
    # datasets = LPDataLoader(None,None,None,None)
    # datasets.count_differ_province(dirs)

    convert_label("/home/hyli/project/OCR/LPRNet_Pytorch/data/CCPD_green_train","./train.txt")






		































# def convert_label(img_path,save_path=None):
#
#     assert img_path,save_path is not None
#
#     img_list = os.listdir(img_path)
#     for img_name in img_list:
#         name_str = img_name.split("_")
#         chinese_str = lp[name_str[-2]]
#         num_str = name_str[-1]
#         chinese_str += num_str
#         print(chinese_str[:-4])
#         # print(num_str)
#
#
#
# if __name__ == '__main__':
#     convert_label("./train")

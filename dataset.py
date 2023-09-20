import os

import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

np.seterr(divide='ignore', invalid='ignore')

import pandas as pd
from sklearn.model_selection import KFold


class quality_dataset(data.Dataset):
    def __init__(self, train=False, val=False, test=False, test_tta=False, all=False, KK=0):
        global train_index, val_index, e_file
        self.train = train
        self.val = val
        self.test = test
        self.path = 'C:/Users/123/OneDrive/桌面/data/B. Image Quality Assessment/'

        if train or val or all:
            self.file = '1. Original Images/a. Training Set/'
            e_file = '2. Groundtruths/a. DRAC2022_ Image Quality Assessment_Training Labels.csv'
        else:
            self.file = 'data/DRAC2022_Testing_Set/B. Image Quality Assessment/1. Original Images/b. Testing Set/'
        self.imgs = []
        img_list = [[] for _ in range(3)]

        if test or test_tta:
            for i in range(len(os.listdir(self.file))):
                x = os.listdir(self.file)[i]
                self.imgs.append([self.file + x, -1, x])

        elif train or val:
            csv_file = pd.read_csv(self.path + e_file)
            self.dict_label = {}
            for index, row in csv_file.iterrows():
                image_id = row['image name']
                rank = int(row['image quality level'])
                img_list[rank].append(image_id)

            for i in range(3):
                # print("CV:",KK)
                kf = KFold(n_splits=5, shuffle=True, random_state=5)
                for kk, (a, b) in enumerate(kf.split(range(len(img_list[i])))):
                    if kk == KK:
                        train_index, val_index = a, b
                        if self.train:
                            print("Grade", i, ':', len(train_index), len(val_index))
                if train:
                    for index in train_index:
                        x = img_list[i][index]
                        self.imgs.append([self.path + self.file + x, i, x])
                else:
                    for index in val_index:
                        x = img_list[i][index]
                        self.imgs.append([self.path + self.file + x, i, x])
        elif all:
            csv_file = pd.read_csv(self.path + e_file)
            for index, row in csv_file.iterrows():
                image_id = row['image name']
                rank = int(row['image quality level'])
                self.imgs.append([self.path + self.file + image_id, rank, image_id])

        data_aug = {
            'brightness': 0.4,  # how much to jitter brightness
            'contrast': 0.4,  # How much to jitter contrast
            'scale': (0.8, 1.2),  # range of size of the origin size cropped
            'ratio': (0.8, 1.2),  # range of aspect ratio of the origin aspect ratio cropped
            'img_size': 384
        }
        if train:
            self.transform = T.Compose([
                T.Resize((640, 640)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomResizedCrop(
                    size=((data_aug['img_size'], data_aug['img_size'])),
                    scale=data_aug['scale'],
                    ratio=data_aug['ratio']
                ),

                T.ColorJitter(
                    brightness=data_aug['brightness'],
                    contrast=data_aug['contrast'],
                ),
                T.ToTensor(),
                # T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # copis pretrained
            ])


        elif val or test or all:
            self.transform = T.Compose([
                T.Resize((data_aug['img_size'], data_aug['img_size'])),
                T.ToTensor(),
                # T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        elif test_tta:
            self.transform = T.Compose([
                T.Resize((640, 640)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomResizedCrop(
                    size=((data_aug['img_size'], data_aug['img_size'])),
                    scale=data_aug['scale'],
                    ratio=data_aug['ratio']
                ),
                T.ToTensor(),
                # T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # copis pretrained
            ])
        print(len(self.imgs))

    def __getitem__(self, index):
        img, label, name = self.imgs[index]
        data = Image.open(img).convert('RGB')
        data = self.transform(data)
        return data, label, name

    def __len__(self):
        return len(self.imgs)
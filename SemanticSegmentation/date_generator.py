from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import torch
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from image_aug import ImageAugment
from config import args


def get_name(root, mode_folder=True):
    for root, dirs, file in os.walk(root):
        if mode_folder:
            return dirs
        else:
            return file


class TotalDataset(Dataset):
    """read all image name and label"""
    def __init__(self, data, transform=None):
        self.all_item = data
        self.transform = transform

    def __len__(self):
        return len(self.all_item)

    def __getitem__(self, item_id):  # generate data when giving index
        image_name = self.all_item[item_id][0]
        label_name = self.all_item[item_id][1]
        image = cv2.imread(image_name)   # cv2.IMREAD_GRAYSCALE
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)

        return sample


class DataDemo(object):
    """  demonstrate one image or label in data set, index is needed  """
    def __init__(self, data, index):
        self.dataset = TotalDataset(data, transform=transforms.Compose([ImageFactory(), ToTensor()]))
        self.index = index

    def show(self, image):  # demonstrate the image
        plt.figure("Image", facecolor='#FFFFFF')
        plt.imshow(image)
        plt.axis('on')
        plt.show()

    def demo_image(self):
        for i in range(len(self.index)):
            sample = self.dataset[self.index[i]]
            print("image_shape", sample["image"].shape)
            self.show(sample["image"])


class Aug(object):
    """class for preprocessing images. """
    def __call__(self, sample):
        wbw = ImageAugment()   # ImageAugment class will augment the img and label at same time
        seq = wbw.aug_sequence()
        image, label = sample["image"], sample["label"]
        image_aug, label_aug = wbw.aug(image, label, seq)
        return {"image": image_aug, "label": label_aug}


class ImageFactory(object):
    """class for preprocessing images"""
    def __init__(self, input_size=args.image_size):
        self.size = input_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        return {"image": image, "label": label}

    def resize(self, image):  # adjust the image size to setting size
        return cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_NEAREST)


class ToTensor(object):
    """change sample to tensor"""
    def __init__(self, demo=False):
        self.demo = demo

    def __call__(self, sample, color=True):
        image, label = sample["image"], sample["label"]
        if not self.demo:
            image = np.transpose(image, (2, 0, 1))  # batch first
            image = torch.from_numpy(image/255)  # convert numpy data to tensor
            label = torch.from_numpy(label)
        return {"image": image, "label": label}


class MakeList(object):
    """
    this class used to make list of data for model train and test, return the root name of each image
    """
    def __init__(self, root):
        self.root_image = root + "image/"
        self.root_label = root + "label/"

    def make_list(self):
        total = []
        folder_list = get_name(self.root_image, mode_folder=False)
        for i in folder_list:
            image_root = self.root_image + i   # record root for each label and relative image
            label_root = self.root_label + i
            total.append([image_root, label_root])
        train_list, val_list = train_test_split(total, random_state=1, train_size=0.9)
        return {"train": train_list, "val": val_list}


class PrepareList(object):
    def __init__(self, root):
        self.root = root

    def generate_list(self):
        names = get_name(self.root, mode_folder=False)
        all_list = []
        for i, data in enumerate(names):
            all_list.append(data)
        return all_list
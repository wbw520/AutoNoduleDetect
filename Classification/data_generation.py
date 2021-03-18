from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tools.tools import make_dict, get_name
from tools.transform_func import make_transform
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class CXRDataset(Dataset):
    """read all image name and label"""
    def __init__(self, data, transform=None):
        self.all_item = data
        self.transform = transform

    def __len__(self):
        return len(self.all_item)

    def __getitem__(self, item_id):  # generate data when giving index
        while not os.path.exists(self.all_item[item_id][0]):
            print("not exist image:" + self.all_item[item_id][0])
            item_id += 1
        image_name = self.all_item[item_id][0]
        label = self.all_item[item_id][1]
        label = torch.from_numpy(np.array(label))
        image = Image.open(image_name)
        if self.transform:
            image = self.transform(image)
        return {"image": image, "label": label, "names": image_name}


class MakeList(object):
    """
    this class used to make list of data for model train and test, return the root name of each image
    root: txt file records condition for every cxr image
    """
    def __init__(self, args, need_part=True):
        self.image_root = args.data_dir
        self.mode = args.mode
        self.dict = self.read_txt()
        self.need_part = need_part
        self.file_name = ["nodule", "nonnodule", "normal"]
        self.ratio = {"nonnodule": args.non_nodule_ratio, "normal": args.normal_ratio}

    def read_txt(self):
        data = pd.read_csv("total_data.csv")
        data = data[["ID", "右肺尖部", "左肺尖部", "右上肺野", "左上肺野", "右中肺野", "左中肺野", "右下肺野", "左下肺野"]].values
        return make_dict(data)

    def get_all_folder(self):
        total_folder = {"train": [], "val": []}
        print("----------------")
        for folder in self.file_name:
            files = get_name(self.image_root+folder)
            if folder != "nodule":
                need, drop = train_test_split(files, random_state=1, train_size=self.ratio[folder])
                train, val = train_test_split(need, random_state=1, train_size=0.85)
                total_folder["train"].append(train)
                total_folder["val"].append(val)
                continue
            train, val = train_test_split(files, random_state=1, train_size=0.85)
            total_folder["train"].append(train)
            total_folder["val"].append(val)
        return total_folder

    def get_wanted_data(self):
        imgs_right = {"train": [], "val": []}
        imgs_left = {"train": [], "val": []}
        total_folder = self.get_all_folder()
        for phase in ["train", "val"]:
            for i in range(len(total_folder[phase])):
                for img in total_folder[phase][i]:
                    label_left = self.deal_label(self.dict[img], part="left")
                    if not (i == 0 and label_left == 0):
                        if self.need_part:
                            part = "/left.png"
                        else:
                            part = ""
                        imgs_left[phase].append([os.path.join(self.image_root, self.file_name[i], img) + part, label_left, self.dict[img]])
                    label_right = self.deal_label(self.dict[img], part="right")
                    if not (i == 0 and label_right == 0):
                        if self.need_part:
                            part = "/right.png"
                        else:
                            part = ""
                        imgs_right[phase].append([os.path.join(self.image_root, self.file_name[i], img) + part, label_right, self.dict[img]])
        if self.mode == "right":
            train = imgs_right["train"]
            val = imgs_right["val"]
            return train, val
        if self.mode == "left":
            train = imgs_left["train"]
            val = imgs_left["val"]
            return train, val
        if self.mode == "all":
            train = imgs_left["train"] + imgs_right["train"]
            val_left = imgs_left["val"]
            val_right = imgs_right["val"]
            return train, val_right, val_left

    def deal_label(self, data, part):
        label = 0
        if part == "right":
            if int(data[0]) + int(data[2]) + int(data[4]) + int(data[6]) > 0:
                label = 1
        else:
            if int(data[1]) + int(data[3]) + int(data[5]) + int(data[7]) > 0:
                label = 1
        return label


class MakeListTest():
    def __init__(self, args):
        self.image_root = args.data_dir
        self.mode = args.mode
        self.dict = self.read_txt()
        self.file_name = "for_test_seg"

    def read_txt(self):
        data = pd.read_csv("test_data.csv")
        data = data[["ID", "右肺尖部", "左肺尖部", "右上肺野", "左上肺野", "右中肺野", "左中肺野", "右下肺野", "左下肺野"]].values
        return make_dict(data)

    def get_files(self):
        folders = get_name(os.path.join(self.image_root, self.file_name))
        total = []
        for imgs in folders:
            total.append([os.path.join(self.image_root, self.file_name, imgs), "test", self.dict[imgs]])
        return total


def load_data(args):
    if args.mode == "all":
        train, val_right, val_left = MakeList(args).get_wanted_data()
        dataset_train = CXRDataset(train, transform=make_transform(args, "train"))
        dataset_val_right = CXRDataset(val_right, transform=make_transform(args, "val"))
        dataset_val_left = CXRDataset(val_left, transform=make_transform(args, "val"))
        data_loader_train = DataLoaderX(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        data_loader_val_right = DataLoaderX(dataset_val_right, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        data_loader_val_left = DataLoaderX(dataset_val_left, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        print("load data " + args.mode + " over")
        return {"train": data_loader_train, "val_right": data_loader_val_right, "val_left": data_loader_val_left}
    else:
        train, val = MakeList(args).get_wanted_data()
        dataset_train = CXRDataset(train, transform=make_transform(args, "train"))
        dataset_val = CXRDataset(val, transform=make_transform(args, "val"))
        data_loader_train = DataLoaderX(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        data_loader_val = DataLoaderX(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        print("load data " + args.mode + " over")
        return {"train": data_loader_train, "val": data_loader_val}
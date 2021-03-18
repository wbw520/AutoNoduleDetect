from random import shuffle
import pandas as pd
from Classification.data_generation import MakeList
from Classification.train import get_args_parser
import argparse


def calculation(data, mode):
    count = 0
    dict = {"nodule": 0, "nonnodule": 0, "normal": 0}
    for img in data:
        if img[1] == 1:
            count += 1
        if "nonnodule" in img[0]:
            dict["nonnodule"] += 1
        elif "normal" in img[0]:
            dict["normal"] += 1
        else:
            dict["nodule"] += 1
    print(mode + "  " + str(count) + "/" + str(len(data)))
    print(mode, dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('for count data', parents=[get_args_parser()])
    args = parser.parse_args()
    args.mode = "right"
    train_right, val_right = MakeList(args).get_wanted_data()
    print("right mode")
    calculation(train_right, "train")
    calculation(val_right, "val")

    args.mode = "left"
    train_left, val_left = MakeList(args).get_wanted_data()
    print("left mode")
    calculation(train_left, "train")
    calculation(val_left, "val")
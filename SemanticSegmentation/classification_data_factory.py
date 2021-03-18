import cv2
import numpy as np
import os
import torch
from tools import ColorTransition


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
conv = ColorTransition()


class DealWith(object):
    def __init__(self, args, resize=False, take_mask=False):
        self.root_image = args.data_deal
        self.resize = resize  # use resize for same size output, or use original resolution
        self.take_mask = take_mask
        self.args = args

    # used to get x coordinate
    def get_x(self, m, start, end):
        rows, cols = m.shape
        x1 = 0
        x2 = 0
        change = 0
        for i in range(cols):
            count = 0
            for j in range(rows):
                if start <= m[j][i] <= end:
                    count = 1
            if change == 0:
                if count != 0:
                    x1 = i
                    change = 1
            else:
                if count == 0:
                    x2 = i
                    if (x2 - x1) < self.args.x_min:
                        change = 0
                        x1 = 0
                        x2 = 0
                        continue
                    break
        if x1 > x2:
            x1 = 0
            x2 = 0
        return x1, x2

    # used to get y coordinate
    def get_y(self, m, x1, x2, start, end):
        rows, cols = m.shape
        sa = abs(x1 - x2)
        y1 = 0
        y2 = 0
        change = 0
        for i in range(rows):
            count = 0
            for j in range(sa):
                if start <= m[i][x1+j] <= end:
                    count = 1
            if change == 0:
                if count != 0:
                    y1 = i
                    change = 1
            else:
                if count == 0:
                    y2 = i
                    if (y2 - y1) < self.args.y_min:
                        change = 0
                        y1 = 0
                        y2 = 0
                        continue
                    break
        if y1 > y2:
            y1 = 0
            y2 = 0
        return y1, y2

    def get_demo(self, name):
        image = cv2.imread(self.root_image + name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pic = cv2.resize(image, (self.args.image_size, self.args.image_size), interpolation=cv2.INTER_NEAREST)
        pic = np.transpose(pic, (2, 0, 1))
        pic = np.array([pic])
        pic = torch.from_numpy(pic/255)
        pic = pic.to(device, dtype=torch.float32)
        return image, pic

    def get_location_mask(self, img, location):
        rows, cols = img.shape
        for i in range(rows):
            for j in range(cols):
                if img[i][j] == location:
                    img[i][j] = location
                else:
                    img[i][j] = 0
        return img.astype(np.uint8)

    def deal_all(self, img,  name):
        """
        all image production for left and right lung
        :param img: mask of current image
        :param name: name of current image
        :return: None
        """
        print(name)
        pic = cv2.imread(self.root_image + name)  # want save image with rgb
        out_right, mask_right, gg_right, coordinate_right = self.write(img, pic, [5, 6, 7, 8])
        out_left, mask_left, gg_left, coordinate_left = self.write(img, pic, [1, 2, 3, 4])
        if gg_right and gg_left:
            os.makedirs(self.args.root_save + name[:-4])
            cv2.imwrite(self.args.root_save + name[:-4] + "/right.png", out_right)
            cv2.imwrite(self.args.root_save + name[:-4] + "/left.png", out_left)
            cv2.imwrite(self.args.root_save + name[:-4] + "/total.png", pic)
            cv2.imwrite(self.args.root_save + name[:-4] + "/right_mask.png", mask_right)
            cv2.imwrite(self.args.root_save + name[:-4] + "/left_mask.png", mask_left)
            cv2.imwrite(self.args.root_save + name[:-4] + "/right_mask_color.png",
                        cv2.cvtColor(conv.id2trainId(mask_right, reverse=True), cv2.COLOR_BGR2RGB))
            cv2.imwrite(self.args.root_save + name[:-4] + "/left_mask_color.png",
                        cv2.cvtColor(conv.id2trainId(mask_left, reverse=True), cv2.COLOR_BGR2RGB))
            hehe = open(self.args.root_save + name[:-4] + "/coordinate.txt", "w")
            wbw_right = "right"+","+str(coordinate_right[0]) + "," + str(coordinate_right[1]) + "," + str(coordinate_right[2]) + "," +str(coordinate_right[3]) + "," +"\n"
            wbw_left = "left"+","+str(coordinate_left[0]) + "," + str(coordinate_left[1]) + "," + str(coordinate_left[2]) + "," +str(coordinate_left[3]) + "," +"\n"
            hehe.write(wbw_right)
            hehe.write(wbw_left)
        else:
            print("---------------------")

    def write(self, img, pic, location):
        gg =True
        x1, x2 = self.get_x(img, location[0], location[-1])  # get coordinate of x
        y1, y2 = self.get_y(img, x1, x2, location[0], location[-1])   # get coordinate of y
        print(x1, x2, y1, y2)
        new_x1 = 0
        new_x2 = 0
        new_y1 = 0
        new_y2 = 0
        img = img[y1:y2, x1:x2].astype(np.uint8)   # get segmented part of mask
        final = 0
        mask = 0
        if location == [5, 6, 7, 8] and x2 > 256:   # drop results of segmentation not satisfy some setting threshold
            gg = False
        if location == [1, 2, 3, 4] and x1 < 256:
            gg = False
        if (x1 - x2) * (y1 - y2) == 0:
            gg = False
        if x1*x2*y1*y2 == 0:
            gg = False
        if gg:
            mask = img
            rows, cols, p = pic.shape
            new_x1 = int(cols * x1 / 512)    # coordination translation to original image
            new_x2 = int(cols * x2 / 512)
            new_y1 = int(rows * y1 / 512)
            new_y2 = int(rows * y2 / 512)
            final = pic[new_y1:new_y2, new_x1:new_x2]
            if self.resize:
                final = cv2.resize(final, (224, 224), interpolation=cv2.INTER_NEAREST)   # resize the output
        return final, mask, gg, [new_x1, new_y1, new_x2, new_y2]  # x_min, y_min, x_max, y_max

    def deal_single(self, img, name, location):
        """
        used for processing one needed location
        img: mask of current image
        name:name of current image
        location: part need segmentation e.g. 6 for right_up
        return: None
        """
        pic = cv2.imread(self.root_image + name)
        out, mask, gg, coordinate = self.write(img, pic, [location])
        if gg:
            cv2.imwrite(self.args.root_save + "selected_part_"+name, out)
        else:
            raise Exception("can not seg this image:")
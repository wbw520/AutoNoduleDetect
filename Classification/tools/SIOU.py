import cv2
import numpy as np
from copy import deepcopy


class Siou():
    def __init__(self, args):
        self.args = args

    def get_x(self, m):
        rows, cols = m.shape
        x1 = 0
        x2 = 0
        change = 0
        for i in range(cols):
            count = 0
            for j in range(rows):
                if m[j][i] != 0:
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
                elif count == 1:
                    x2 = i
        if x1 > x2:
            x1 = 0
            x2 = 0
        return x1, x2

    def get_y(self, m, x1, x2):
        rows, cols = m.shape
        sa = abs(x1 - x2)
        y1 = 0
        y2 = 0
        change = 0
        for i in range(rows):
            count = 0
            for j in range(sa):
                if m[i][x1+j] != 0:
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
                elif count == 1:
                    y2 = i
        if y1 > y2:
            y1 = 0
            y2 = 0
        return y1, y2

    def search(self, img):  # calculate coordinate for attention area
        x1, x2 = self.get_x(img)
        y1, y2 = self.get_y(img, x1, x2)
        have = True
        if (x1 + x2 + y1 + y2) != 0:
            for i in range(x1, x2+1):
                for j in range(y1, y2+1):
                    img[j][i] = 0    # black the found area for next search
        else:
            have = False
        return [x1, y1, x2, y2], img, have

    def search2(self, img):  # for calculate union of attention and lung field
        x1, x2 = self.get_x(img)
        y1, y2 = self.get_y(img, x1, x2)
        return [x1, y1, x2, y2]

    def deep_search(self, img):
        # print("start deep search")
        # have means whether get the attention box in one search
        coordinate = []
        have = True
        while have:
            coordi, img, have = self.search(img)
            if have:
                coordinate.append(coordi)
        # print("deep search finish")
        return coordinate

    def location_discriminate(self, location):
        location_have = []
        if self.args.mode == "right":
            LL = [0, 2, 4, 6]   # location of lung field
            PP = [5, 6, 7, 8]   # label value
            for i in range(len(LL)):
                if location[LL[i]] != "0":
                    location_have.append(PP[i])
        elif self.args.mode == "left":
            LL = [1, 3, 5, 7]
            PP = [4, 3, 2, 1]
            for i in range(len(LL)):
                if location[LL[i]] != "0":
                    location_have.append(PP[i])
        return location_have

    def siou(self, att, mask, condition, coordinate):
        """
        att: attention map
        mask: segmented results
        condition: responding lung filed value
        coordinate: coordinate of all found attention area
        return: whether satisfied setting threshold and coordinate of union
        """
        rows, cols = att.shape
        mask = cv2.resize(mask, (cols, rows), interpolation=cv2.INTER_NEAREST)
        self_count = 0
        iou_count = 0
        have = False
        new_coordinate = 0
        PP = np.zeros((rows, cols), dtype="int")  # for saving union area
        for i in range(coordinate[0], coordinate[2]+1):
            for j in range(coordinate[1], coordinate[3]+1):
                if att[j][i] == 255:
                    self_count += 1
                    if mask[j][i] in condition:
                        iou_count += 1
                        PP[j][i] = 255
        standard = iou_count/(self_count+1)
        # print("siou for this patch", standard)
        if standard > self.args.SI_standard:
            have = True
            new_coordinate = self.search2(PP)   # get the coordinate of union
        return have, new_coordinate

    def deal_siou(self, root, att_map, label):
        correct_coordinate = []
        location_condition = self.location_discriminate(label)  # confirm lung filed with disease
        mask = cv2.imread(root + "/" + self.args.mode + "_mask.png", cv2.IMREAD_GRAYSCALE)
        for_search = deepcopy(att_map)
        coordinate = self.deep_search(for_search)
        for i in range(len(coordinate)):
            outcome, new_coordinate = self.siou(att_map, mask, location_condition, coordinate[i])
            if outcome:
                correct_coordinate.append(new_coordinate)
        return correct_coordinate  # a list of all boxes's coordinate


class CalFinalCoordinate():
    def __init__(self, args):
        self.mode = args.mode
        self.args = args

    def read_txt_coordinate(self, name):
        with open(name + "/coordinate.txt", "r", encoding="UTF-8") as data:
            name = []
            lines = data.readlines()
            for line in lines:
                a = line.split(",")
                name.append(a)
            if self.mode == "right":   # we record first right and then left in segmentation part
                pp = 0
            else:
                pp = 1
            return name[pp]

    def translate_coordinate(self, zuobiao, target, orl):
        # it will calculate responding coordinate for original image
        rows_orl, cols_orl = orl.shape
        rows_target, cols_target = target.shape
        new_x1 = int(cols_target * zuobiao[0] / cols_orl)
        new_x2 = int(cols_target * zuobiao[2] / cols_orl)
        new_y1 = int(rows_target * zuobiao[1] / rows_orl)
        new_y2 = int(rows_target * zuobiao[3] / rows_orl)
        return [new_x1, new_y1, new_x2, new_y2]

    def total_image_coordinate(self, coordinate_seg_att, coordinate_seg_box):
        # pan for coordinate in original image with recorded inf in segmentation part
        coordinate_seg_att[0] += int(coordinate_seg_box[1])
        coordinate_seg_att[1] += int(coordinate_seg_box[2])
        coordinate_seg_att[2] += int(coordinate_seg_box[1])
        coordinate_seg_att[3] += int(coordinate_seg_box[2])
        return coordinate_seg_att

    def coordinate(self, name, att_coordinate):
        # att_coordinate is calculated by siou part
        mask = cv2.imread(name + "/" + self.args.mode + "_mask.png", cv2.IMREAD_GRAYSCALE)
        seg_image = cv2.imread(name + "/" + self.args.mode + ".png", cv2.IMREAD_GRAYSCALE)

        seg_box_coordinate = self.read_txt_coordinate(name)  # get relative coordinate
        seg_att_coordinate = self.translate_coordinate(att_coordinate, seg_image, mask)
        seg_att_coordinate_copy = deepcopy(seg_att_coordinate)
        total_image_coordi = self.total_image_coordinate(seg_att_coordinate_copy, seg_box_coordinate)
        return total_image_coordi, seg_att_coordinate


        # xml_size = np.zeros((1000, 1000), dtype="int")
        # xml_coordinate = self.translate_coordinate(total_image_coordi, xml_size, total_image)
        # self.write_txt(xml_coordinate, name)
        # default resize [1000, 1000] for faster-rcnn
        # xml_image = cv2.resize(total_image, (1000, 1000), interpolation=cv2.INTER_LINEAR)
        # default jpg for faster-rcnn
        # cv2.imwrite(C.xml_root + name + ".jpg", xml_image)
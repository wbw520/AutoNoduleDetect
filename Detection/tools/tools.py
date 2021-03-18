from pycocotools.coco import COCO
import cv2
import pandas as pd
from tools.make_json import make_json
from tqdm.auto import tqdm


def iou(box1, box2, mode, thresh):
    '''
    两个框（二维）的 iou 计算

    注意：边框以左上为原点
    box1 predict  box2 man made
    box:[Xmin, Ymin, Xmax, Ymax]
    '''
    output = False
    in_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    sbox1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    sbox2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = sbox1 + sbox2 - inter
    iou = inter / union
    box1_self = inter / sbox1
    box2_self = inter / sbox2
    if mode == "predict" and box1_self >= thresh:
        output = True
    elif mode == "man" and box2_self >= thresh:
        output = True
    elif mode == "iou" and iou >= thresh:
        output = True
    return output


def make_coordinate_trans(box):
    return [box[0], box[1], box[0]+box[2], box[1]+box[3]]


def read_coco(root, use_trans=True, test=False):
    record = {}
    coco = COCO(root)
    img_ids = list(sorted(coco.imgs.keys()))
    total = 0
    ccc = 0
    print(len(img_ids))
    for id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=id)
        path = coco.loadImgs(id)[0]['file_name']
        target = coco.loadAnns(ann_ids)
        box_list = []
        for box in target:
            current_box = box["bbox"]
            if use_trans:
                current_box = make_coordinate_trans(current_box)
            box_list.append(current_box)
        if len(box_list) == 0:
            ccc += 1
        if test:
            record.update({path[:-4]: box_list})
        else:
            record.update({path: box_list})
        total += len(box_list)
    print(total)
    print(ccc)
    return record


def cal_area(box):
    area = int((box[2]-box[1])*(box[3]-box[1]))
    return area


def box_cal(box_pre, box_true, thresh, mode):
    record_true = {}
    record_pre = {}
    area_size = 0
    area_true = 0
    for i in range(len(box_pre)):
        for j in range(len(box_true)):
            if i == 0:
                area_true += cal_area(box_true[j])
            if iou(box_pre[i], box_true[j], mode, thresh):
                if i not in record_pre:
                    record_pre.update({i: "good_pre"})
                    area_size += cal_area(box_pre[i])
                if j not in record_true:
                    record_true.update({j: "good_true"})
    found_true_num = len(record_true)
    found_pre_num = len(record_pre)
    return found_pre_num, found_true_num, area_size, area_true


def cal_overlap(D1, D2, thresh, mode):
    # D2 is the truth
    count_true = 0
    count_pred = 0
    found_true = 0
    found_pre = 0
    area = 0
    area_true = 0
    for v, k in D2.items():
        true_boxes_per = k
        if v not in D1:
            continue
        pre_boxes_per = D1[v]
        count_true += len(true_boxes_per)
        count_pred += len(pre_boxes_per)
        current_found_pre, current_found_true, current_area, current_area_true = box_cal(pre_boxes_per, true_boxes_per, thresh, mode)
        found_true += current_found_true
        found_pre += current_found_pre
        area += current_area
        area_true += current_area_true
    recall = found_true/count_true
    precision = found_pre/count_pred
    F1 = (2*precision*recall)/(precision + recall)
    avg_area = area//found_pre
    print(avg_area)
    print(area_true//count_true)
    print(count_true)
    print(count_pred)
    print(found_true)
    print(found_pre)
    print("recall: ", round(recall, 3), "precision: ", round(precision, 3), "F1: ", round(F1, 3))


def make_dict(input):
    AD = {}
    for i in range(len(input)):
        AD.update({input[i][0]: input[i][1:]})
    return AD


class Filter():
    def __init__(self, root, for_test=False, name_list=None):
        self.iou = 2
        self.overlap_standard = 0.7
        self.root = root
        self.dict = self.read_csv()
        self.add_count = 0
        self.all_count = 0
        self.new_record = {}
        self.for_test = for_test
        self.name_list = name_list

    def read_csv(self):
        data = pd.read_csv("test_data.csv")
        data = data[["ID", "右肺尖部", "左肺尖部", "右上肺野", "左上肺野", "右中肺野", "左中肺野", "右下肺野", "左下肺野"]].values
        return make_dict(data)

    def read_txt_coordinate(self, name):
        with open(name + "/coordinate.txt", "r", encoding="UTF-8") as data:
            coordinate = []
            lines = data.readlines()
            for line in lines:
                a = line.split(",")
                coordinate.append(list(map(float, a[1:-1])))
            # right coordinate first
            return coordinate

    def factory(self, last_record, current_pre, coco_name):
        for v, k in tqdm(last_record.items()):
            name = v
            if self.for_test and name not in self.name_list:
                print("-----------")
                continue
            boxes_last = k
            self.new_record.update({name: []})
            boxes_current = current_pre[v]
            self.all_count += len(boxes_current)
            for i, current in enumerate(boxes_current):
                over_index = False
                for j in range(len(boxes_last)):
                    if iou(current, boxes_last[j], mode="iou", thresh=self.iou):
                        over_index = True
                if not over_index:
                    self.mask_overlap(name, current)
            if len(self.new_record[name]) == 0:
                self.new_record.pop(name)
        print("add all boxes num: ", self.all_count)
        print("add new boxes num: ", self.add_count)
        if not self.for_test:
            make_json(self.new_record, coco_name, self.root)
        else:
            return self.new_record

    def translate_coordinate(self, zuobiao, target, orl):
        # it will calculate responding coordinate for original image
        rows_orl, cols_orl = (orl[3]-orl[1]), (orl[2]-orl[0])
        rows_target, cols_target = target.shape
        new_x1 = int(cols_target * zuobiao[0] / cols_orl)
        new_x2 = int(cols_target * zuobiao[2] / cols_orl)
        new_y1 = int(rows_target * zuobiao[1] / rows_orl)
        new_y2 = int(rows_target * zuobiao[3] / rows_orl)
        return [new_x1, new_y1, new_x2, new_y2]

    def loaction_search(self, coor_seg, coor2):
        rr = "no"
        right_coor = coor_seg[0]
        left_coor = coor_seg[1]
        if iou(right_coor, coor2, mode="man", thresh=1):
            rr = "right"
            return rr, [(coor2[0]-right_coor[0]), (coor2[1]-right_coor[1]), (coor2[2]-right_coor[0]), (coor2[3]-right_coor[1])]
        elif iou(left_coor, coor2, mode="man", thresh=1):
            rr = "left"
            return rr, [(coor2[0]-left_coor[0]), (coor2[1]-left_coor[1]), (coor2[2]-left_coor[0]), (coor2[3]-left_coor[1])]
        else:
            return rr, []

    def location_discriminate(self, location):
        location_have = []
        LL = [0, 2, 4, 6, 1, 3, 5, 7]   # location of lung field
        PP = [5, 6, 7, 8, 4, 3, 2, 1]   # label value
        for i in range(len(LL)):
            if location[LL[i]] != 0:
                location_have.append(PP[i])
        return location_have

    def overlap(self, mask, condition, coordinate):
        """
        att: attention map
        mask: segmented results
        condition: responding lung filed value
        coordinate: coordinate of all found attention area
        return: whether satisfied setting threshold and coordinate of union
        """
        overlap_count = 0
        have = False
        for i in range(coordinate[0], coordinate[2]+1):
            for j in range(coordinate[1], coordinate[3]+1):
                if mask[j][i] in condition:
                    overlap_count += 1
        standard = overlap_count/((coordinate[2] - coordinate[0]) * (coordinate[3] - coordinate[1]))
        if standard > self.overlap_standard:
            have = True
        return have

    def mask_overlap(self, name, box):
        mask_right = cv2.imread(self.root + "/" + name + "/" + "right_mask.png", cv2.IMREAD_GRAYSCALE)
        mask_left = cv2.imread(self.root + "/" + name + "/" + "left_mask.png", cv2.IMREAD_GRAYSCALE)
        coordinate = self.read_txt_coordinate(self.root+name)
        loaction, new_box = self.loaction_search(coordinate, box)
        loaction_label = self.location_discriminate(self.dict[name])
        add = False
        if loaction == "right":
            new_new_box = self.translate_coordinate(new_box, mask_right, coordinate[0])
            add = self.overlap(mask_right, loaction_label, new_new_box)
        if loaction == "left":
            new_new_box = self.translate_coordinate(new_box, mask_left, coordinate[1])
            add = self.overlap(mask_left, loaction_label, new_new_box)
        if add:
            self.new_record[name].append(box)
            self.add_count += 1
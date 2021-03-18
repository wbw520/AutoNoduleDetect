from pycocotools.coco import COCO


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
    if mode == "predict" and box1_self > thresh:
        output = True
    elif mode == "man" and box2_self > thresh:
        output = True
    elif mode == "iou" and iou > thresh:
        output = True
    return output


def make_coordinate_trans(box):
    return [box[0], box[1], box[0]+box[2], box[1]+box[3]]


def read_coco(root, use_trans=True):
    record = {}
    coco = COCO(root)
    img_ids = list(sorted(coco.imgs.keys()))
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
        record.update({path[:-4]: box_list})
    return record


def box_cal(box_pre, box_true):
    record_true = {}
    record_pre = {}
    for i in range(len(box_pre)):
        for j in range(len(box_true)):
            if iou(box_pre[i], box_true[j], "man", 0.1):
                if i not in record_pre:
                    record_pre.update({i: "good_pre"})
                if j not in record_true:
                    record_true.update({j: "good_true"})
    found_true_num = len(record_true)
    found_pre_num = len(record_pre)
    return found_pre_num, found_true_num


def cal_overlap(D1, D2):
    # D2 is the truth
    count_true = 0
    count_pred = 0
    found_true = 0
    found_pre = 0
    for v, k in D2.items():
        true_boxes_per = k
        pre_boxes_per = D1[v]
        count_true += len(true_boxes_per)
        count_pred += len(pre_boxes_per)
        current_found_pre, current_found_true = box_cal(pre_boxes_per, true_boxes_per)
        found_true += current_found_true
        found_pre += current_found_pre
    precision = found_true/count_true
    recall = found_pre/count_pred
    F1 = (2*precision*recall)/(precision + recall)
    print(count_true)
    print(count_pred)
    print("recall: ", round(recall, 3), "precision: ", round(precision, 3), "F1: ", round(F1, 3))




# test_boxes = read_coco("/home/wbw/PycharmProjects/faster/instances_test.json")
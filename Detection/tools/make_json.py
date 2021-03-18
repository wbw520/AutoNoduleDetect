import cv2
import json
import os


def make_json(data, phase, root_all_image):
    class_index = {1: "nodule"}
    dataset = {"info": {"year": 2020, "version": "2020", "description": "for_cxr", "contributor": "wbw", "url": "", "date_created": "2020.08.09"},
               "license": {"id": 1, "url": "", "name": "wangbowen"},
               "images": [],
               "annotations": [],
               "categories": []}
    total_bbox_id = 0
    for s, k in enumerate(list(class_index.keys())):
        dataset["categories"].append({"id": k, "name": class_index[k]})

    for i, key in enumerate(data.keys()):
        if i % 100 == 0:
            print(str(i) + "/" + str(len(data)))
        # if not os.path.exists(root_all_image+"nodule/"+key):
        #     print(root_all_image+key)
        #     continue
        current_img = cv2.imread(os.path.join(root_all_image, key + "/total.png"))
        height, weight, c = current_img.shape
        dataset["images"].append({"license": 1,
                                  "file_name": key,
                                  "id": i,
                                  "weight": weight,
                                  "height": height})
        box = data[key]
        for j in range(len(box)):
            w = box[j][2] - box[j][0]
            h = box[j][3] - box[j][1]
            dataset["annotations"].append({"area": w*h,
                                           "bbox": [box[j][0], box[j][1], w, h],
                                           "category_id": 1,
                                           "id": total_bbox_id,
                                           "image_id": i,
                                           "iscrowd": 0,
                                           "segmentation": [[]]})
            total_bbox_id += 1
    with open("cxr_" + phase + ".json", "w") as f:
        json.dump(dataset, f)
import pandas as pd
import os
import shutil
import csv


def dict_factory(data):
    D = {}
    for i in range(len(data)):
        D.update({data[i][0]+"_"+str(data[i][1]): data[i][3:]})
    return D


def get_name(root, mode_folder=True):
    for root, dirs, file in os.walk(root):
        if mode_folder:
            return dirs
        else:
            return file


def translate(folder_name, folder_root, D=None, normal=False):
    folder_record = []
    out_list = []
    for i in range(len(folder_name)):
        if i % 500 == 0:
            print(str(i)+"/"+str(len(folder_name)))
        if D:
            current_name = folder_name[i][:20]
            if current_name in folder_record:
                continue
            folder_record.append(current_name)
            out_list.append([current_name] + list(D[current_name]))
            if os.path.exists(root_final_nodule+current_name):
                continue
            shutil.copytree(folder_root+folder_name[i], root_final_nodule+current_name)
            print([current_name] + list(D[current_name]))
        else:
            if not normal:
                root_final = root_final_nonnodule
                current_name = folder_name[i][:20]
                if current_name in folder_record:
                    continue
            else:
                root_final = root_final_normal
                current_name = folder_name[i]
            folder_record.append(current_name)
            out_list.append([current_name] + [0, 0, 0, 0, 0, 0, 0, 0])
            if os.path.exists(root_final+current_name):
                continue
            shutil.copytree(folder_root+folder_name[i], root_final+current_name)
    return out_list


def make_csv(data, name):
    f_val = open(name + ".csv", "w", encoding="utf-8")
    csv_writer = csv.writer(f_val)
    csv_writer.writerow(["ID", "右肺尖部", "左肺尖部", "右上肺野", "左上肺野", "右中肺野", "左中肺野", "右下肺野", "左下肺野"])
    for i in range(len(data)):
        csv_writer.writerow(data[i])
    f_val.close()


def pair(input1):
    nodule_D = dict_factory(input1)
    nodule_folder = get_name(image_root_nodule)
    non_nodule_folder = get_name(image_root_nonnodule)
    normal_folder = get_name(image_root_normal)
    nodule_list = translate(nodule_folder, image_root_nodule, nodule_D)
    non_nodule_list = translate(non_nodule_folder, image_root_nonnodule)
    normal_list = translate(normal_folder, image_root_normal, normal=True)
    final_list = nodule_list+non_nodule_list+normal_list
    make_csv(final_list, "total_data")


if __name__ == '__main__':
    root_final_nodule = "/home/wbw/PAN/final/nodule/"
    root_final_nonnodule = "/home/wbw/PAN/final/nonnodule/"
    root_final_normal = "/home/wbw/PAN/final/normal/"
    for floder in [root_final_nodule, root_final_nonnodule, root_final_normal]:
        if not os.path.exists(floder):
            os.makedirs(floder)
    image_root_nodule = "/media/wbw/HD-GDU3/cxr/nodule_save/"
    image_root_nonnodule = "/media/wbw/HD-GDU3/cxr/non_nodule_save/"
    image_root_normal = "/media/wbw/HD-GDU3/cxr//normal_save/"
    nodule = pd.read_csv("nodule_selected.csv", dtype=str)
    nodule_data = nodule[["CRID", "xdate", "所見", "右肺尖部", "左肺尖部", "右上肺野", "左上肺野", "右中肺野", "左中肺野", "右下肺野", "左下肺野"]].values
    pair(nodule_data)
    # print(get_name(image_root_nodule))
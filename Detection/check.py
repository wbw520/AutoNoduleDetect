import os
import xml.dom.minidom
from tools.make_json import make_json


def get_name(root, mode_folder=True):
    for root, dirs, file in os.walk(root):
        if mode_folder:
            return dirs
        else:
            return file


def read_xml(id):
    dom = xml.dom.minidom.parse("/home/wbw/PAN/final_cxr/ann_xml/"+id)
    name = dom.getElementsByTagName('name')
    x_min = dom.getElementsByTagName('xmin')
    y_min = dom.getElementsByTagName('ymin')
    x_max = dom.getElementsByTagName('xmax')
    y_max = dom.getElementsByTagName('ymax')
    if len(name) == 0:
        print(id)
    for i in range(len(name)):
        a = name[i].firstChild.data
        b = x_min[i].firstChild.data
        c = y_min[i].firstChild.data
        d = x_max[i].firstChild.data
        e = y_max[i].firstChild.data
        if id[:-4] not in D:
            D.update({id[:-4]: [[int(b), int(c), int(d), int(e)]]})
        else:
            D[id[:-4]].append([int(b), int(c), int(d), int(e)])


if __name__ == '__main__':
    D = {}
    files = get_name("/home/wbw/PAN/final_cxr/ann_xml/", mode_folder=False)
    print(len(files))
    for name in files:
        read_xml(name)
    print(len(D))
    # make_json(D, {1: "nodule"}, "train", "/home/wbw/PAN/final_cxr/for_test/")



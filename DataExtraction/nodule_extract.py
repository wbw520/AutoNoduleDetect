# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import csv


def nodule(input):
    x = input
    rr = []  # record the unique combination for cxr image with id and date
    final = []  # record the total information for one image

    for i in range(len(x)):
        if x[i][2] == "結節影" or x[i][2] == "腫瘤影":
            if any(x[i][3:] != np.array([0, 0, 0, 0, 0, 0, 0, 0])):
                if x[i][0]+str(x[i][1]) not in rr:
                    rr.append(x[i][0]+str(x[i][1]))
                    final.append(x[i])
                else:
                    index = rr.index(x[i][0]+str(x[i][1]))
                    if any(final[index][3:] != x[i][3:]):
                        final[index][3:] = list(np.array(final[index][3:]) + np.array(x[i][3:]))  # label merge
                        final[index][3:] = np.array([(1 if a > 0 else 0) for a in final[index][3:]])  # adjust label made all 1
    return rr, final


def non_nodule(in1, in2):
    x = in1
    rr = []  # record the unique combination for cxr image with id and date
    final = []  # record the total information for one image
    for i in range(len(x)):
        if x[i][2] != "結節影" and x[i][2] != "腫瘤影":
            if x[i][0]+str(x[i][1]) not in rr and x[i][0]+str(x[i][1]) not in in2:  # drop patient once have nodule
                rr.append(x[i][0]+str(x[i][1]))
                x[i][3:] = np.array([0, 0, 0, 0, 0, 0, 0, 0])
                final.append(x[i])
    return final


def make_csv(data, name):
    f_val = open(name + ".csv", "w", encoding="utf-8")
    csv_writer = csv.writer(f_val)
    csv_writer.writerow(["CRID", "xdate", "所見", "右肺尖部", "左肺尖部", "右上肺野", "左上肺野", "右中肺野", "左中肺野", "右下肺野", "左下肺野"])
    for i in range(len(data)):
        csv_writer.writerow(data[i])
    f_val.close()


if __name__ == '__main__':
    data_all = pd.read_csv('positive_findings.csv')
    wbw = data_all[["CRID", "xdate", "所見", "右肺尖部", "左肺尖部", "右上肺野", "左上肺野", "右中肺野", "左中肺野", "右下肺野", "左下肺野"]]
    wbw = wbw.values
    nodule_combination, nodule_inf = nodule(wbw)
    non_nodule_inf = non_nodule(wbw, nodule_combination)  # do not allow people once have nodule
    make_csv(nodule_inf, "nodule_selected111")
    make_csv(non_nodule_inf, "non_nodule_selected")
# -*- coding: UTF-8 -*-
import pandas as pd
import os
import pydicom
import shutil


def get_name(root, mode_folder=True):
    for root, dirs, file in os.walk(root):
        if mode_folder:
            return dirs
        else:
            return file


def panduan(a):   # whether there is the file
    B = os.path.exists(a)
    return B


def loadFileInformation(filename):
    information = {}
    ds = pydicom.read_file(filename, force=True)
    information['PatientID'] = ds.PatientID
    information['PatientName'] = ds.PatientName
    information['PatientBirthDate'] = ds.PatientBirthDate
    information['PatientSex'] = ds.PatientSex
    information['StudyID'] = ds.StudyID
    information['StudyDate'] = ds.StudyDate
    information['StudyTime'] = ds.StudyTime
    return information


def reform(md, year):
    if "2012" in str(year):
        k = md.split('-')
        hh = 'CR-0' + k[1]
        return hh
    else:
        return md


def deal_dicom(dicom_root, date, t_id):
    inf = loadFileInformation(dicom_root)
    id = reform(t_id, date)
    if date in inf['StudyDate']:
        if id in inf['PatientID']:
            kk = 0
            pp = root_save+t_id+"_"+inf['StudyDate']+"_" + str(kk) + ".dcm"
            while panduan(pp):
                kk = kk+1
                pp = root_save + t_id+"_"+inf['StudyDate']+"_" + str(kk) + ".dcm"
            print(pp)
            shutil.copy(dicom_root, pp)


def search_copy(data):
    for i in range(len(data)):
        if i % 100 == 0:
            print(str(i) + "/" + str(len(data)))
        CRID, date = data[i][0], str(data[i][1])
        year, md = date[:4], date[4:]
        if int(year) != 2017:
            continue
        year_root = image_root + str(year) + year_suffix + "/"
        if not panduan(year_root):
            continue
        folder_structure_0 = get_name(year_root)[0].split('^')
        folder_structure_1 = folder_structure_0[0] + "^" + folder_structure_0[1] + "^"
        patient_folder = year_root + folder_structure_1 + reform(CRID, year) + "/"
        if not panduan(patient_folder):
            print(patient_folder)
            continue
        dicom_list = get_name(patient_folder, mode_folder=False)
        for j in range(len(dicom_list)):
            current_dicom_root = patient_folder + dicom_list[j]
            deal_dicom(current_dicom_root, date, CRID)


if __name__ == '__main__':
    image_root = "/mnt/image/"
    root_save = "/home/elsa/xia3/"
    year_suffix = "年全画像(胸部正面CR)"
    file_name = "non_nodule_selected.csv"
    data_all = pd.read_csv(file_name)
    wbw = data_all[["CRID", "xdate", "所見", "右肺尖部", "左肺尖部", "右上肺野", "左上肺野", "右中肺野", "左中肺野", "右下肺野", "左下肺野"]]
    wbw = wbw.values
    search_copy(wbw)
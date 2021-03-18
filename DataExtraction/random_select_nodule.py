import os
import random
import shutil


def get_name(root, mode_folder=True):
    for root, dirs, file in os.walk(root):
        if mode_folder:
            return dirs
        else:
            return file


root = "/home/wbw/PAN/final_cxr/for_test/"
root_save = "/home/wbw/PAN/final_cxr/need_annotation/"
all_file = get_name(root)
record = []
# saved = []
# while len(saved) < 500:
#     seed = random.randint(0, len(all_file))
#     if all_file[seed][:11] not in record:
#         print(all_file[seed][:11])
#         saved.append(all_file[seed])
#         all_file.pop(seed)

for i in range(len(all_file)):
    shutil.copy(root+all_file[i]+"/total.png", root_save+all_file[i]+".png")
# for name in saved:
#     shutil.move(root+name, root_save+name)
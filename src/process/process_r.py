import numpy as np
import os
from numpy.random import randint
import shutil


data_path = r'D:\Study\mediaeval2016_test_image\MediaEval2016_test'
sub_dirs = [x[0] for x in os.walk(data_path)]
sub_dirs = sub_dirs[1:]

for sub_dir in sub_dirs:

    store_path = sub_dir.replace(r'D:\Study\mediaeval2016_test_image\MediaEval2016_test', r'D:\Study\input_test_image')
    if os.path.exists(store_path):
        print(store_path)
        continue
    os.makedirs(store_path, exist_ok=True)

    files_list = os.listdir(sub_dir)
    files_list.sort(key=lambda x: int(x[5:-4]))
    files_list = [os.path.join(sub_dir, i) for i in files_list]

    k = 8
    average_duration = (len(files_list)) // k
    offsets = np.multiply(list(range(k)), average_duration) + randint(average_duration, size=k)
    for i in range(len(offsets)):
        src = files_list[offsets[i]]
        print(src)
        dst = os.path.join(store_path, "image" + str(i) + ".jpg")
        print(dst)
        shutil.copy(src, dst)

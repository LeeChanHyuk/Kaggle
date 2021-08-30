import os
<<<<<<< HEAD

# The folder path that i want to change
file_path = 'D:/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
file_names = os.listdir(file_path)
dirs = ['FLAIR', 'T1', 'T1CE', 'T2', 'SEG']
file_name_sort = ['FLAIR0.jpg', 'SEG1.jpg', 'T1CE3.jpg','T12.jpg','T24.jpg']
for folder_name in file_names: # folder name
    for sort in dirs:
        for file in os.listdir(os.path.join(file_path, folder_name, sort)):
            os.remove(os.path.join(file_path,folder_name,sort,file))
        os.rmdir(os.path.join(file_path, folder_name,sort))
    #for sort in file_name_sort:
    #    os.remove(os.path.join(file_path, folder_name, sort))
=======
import json
import glob
import random
import collections
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

local_path = 'D:/rsna-miccai-brain-tumor-radiogenomic-classification' # dataset path
train_df = pd.read_csv(local_path + '/train_labels.csv')

# 결국에는, 환자 한 폴더당 3개의 종류에 따라 Normalized된 이미지를 불러서, 3개를 각각 채널로 삼아서 Network의 Input으로 설정하고
# Inference 또한 똑같은 방식으로 진행한다.
def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

def visualize_sample(
    brats21id,
    slice_i,
    mgmt_value,
    types=("FLAIR", "T1w", "T1wCE", "T2w")
): # 입력으로 들어온 sample을 visualization 해주는 함수
    plt.figure(figsize=(16, 5))
    patient_path = os.path.join(
        local_path,'train/',
        str(brats21id).zfill(5),
    ) # 입력으로 들어온 patient number의 path를 만들어준다
    for i, t in enumerate(types, 1):
        t_paths = sorted(
            glob.glob(os.path.join(patient_path, t, "*")),
            key=lambda x: int(x[:-4].split("-")[-1]),
        ) # 해당 folder의 파일들을 모두 가져와서 sorting 한다.
        data = load_dicom(t_paths[int(len(t_paths) * slice_i)])
        plt.subplot(1, 4, i)
        plt.imshow(data, cmap="gray")
        plt.title(f"{t}", fontsize=16)

    plt.suptitle(f"MGMT_value: {mgmt_value}", fontsize=16)
    plt.show()

import nibabel as nib

seg_path = 'D:/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'

index = 0

def make_name(str):
    if 'flair' in str:
        return 'FLAIR'
    elif 'seg' in str:
        return 'SEG'
    elif 't1ce' in str:
        return 'T1CE'
    elif 't1' in str:
        return 'T1'
    elif 't2' in str:
        return 'T2'

def load_nib():
    for folder in os.listdir(seg_path):
        for index, files in enumerate(os.listdir(os.path.join(seg_path,folder))):
            print(index)
            file_path = os.path.join(seg_path,folder)
            os.makedirs(file_path+'/'+make_name(files),exist_ok=True)
            proxy = nib.load(os.path.join(seg_path,folder,files))
            header = proxy.header
            arr = proxy.get_fdata()
            for z in range(arr.shape[2]):
                cv2.imwrite(file_path+'/'+make_name(files)+'/'+make_name(files)+str(z)+'.jpg',arr[:,:,z])

def load_dicom_file(index):
    _brats21id = train_df.iloc[index]["BraTS21ID"]
    _mgmt_value = train_df.iloc[index]["MGMT_value"]
    visualize_sample(brats21id=_brats21id, mgmt_value=_mgmt_value, slice_i=0.5)
    index += 1
    load_nib(index)

load_nib()
>>>>>>> 4f9ceabbe69081da49aea3517768d00108288580

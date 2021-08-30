import numpy as np
import pydicom
import os
import cv2
from pydicom.pixel_data_handlers.util import apply_voi_lut

local_path = 'D:/rsna-miccai-brain-tumor-radiogenomic-classification' # dataset path
train_folder_path = os.path.join(local_path, 'test')
test_folder_path = os.path.join(local_path, 'test')

save_local_path = 'D:/Task2_dataset_png'

def load_dicom(path, preprocessing = 'norm'):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    if preprocessing == 'norm':
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / (np.max(data) - np.min(data))
        data = (data * 255).astype(np.uint8)
    if preprocessing == 'apply_voi_lut':
        data = apply_voi_lut(dicom.pixel_array, dicom)
    return data


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

sorts = ['FLAIR', 'T1w', 'T1wCE', 'T2w']

for patient_folder in os.listdir(train_folder_path):
    os.makedirs(os.path.join(save_local_path,'apply_voi_lut',patient_folder))
    for sort in sorts:
        os.makedirs(os.path.join(save_local_path,'apply_voi_lut',patient_folder,sort))
        for dicom_file in os.listdir(os.path.join(train_folder_path,patient_folder,sort)):
            img = load_dicom(os.path.join(train_folder_path,patient_folder,sort,dicom_file),'apply_voi_lut')
            cv2.imwrite(os.path.join(save_local_path,'apply_voi_lut',patient_folder,sort,dicom_file[0:len(dicom_file)-4])+'.png', img)

for patient_folder in os.listdir(train_folder_path):
    os.makedirs(os.path.join(save_local_path,'norm',patient_folder))
    for sort in sorts:
        os.makedirs(os.path.join(save_local_path,'norm',patient_folder,sort))
        for dicom_file in os.listdir(os.path.join(train_folder_path,patient_folder,sort)):
            img = load_dicom(os.path.join(train_folder_path,patient_folder,sort,dicom_file))
            cv2.imwrite(os.path.join(save_local_path,'norm',patient_folder,sort,dicom_file[0:len(dicom_file)-4])+'.png', img)

import os

task1_data_path = 'D:/RSNA_ASNR_MICCAI_BraTS2021_ValidationData'
task2_data_path = 'D:/rsna-miccai-brain-tumor-radiogenomic-classification/test'
task1_data_name=[]
task2_data_name=[]
for i in os.listdir(task1_data_path):
    task1_data_name.append(i[len(i)-5:len(i)])
for i in os.listdir(task2_data_path):
    task2_data_name.append(i)
for i in range(len(task2_data_name)):
    if task2_data_name[i] not in task1_data_name:
        print(task2_data_name[i] + "is not true")
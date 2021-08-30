import os
import cv2

data_path = 'D:/Testing_data_png'
seg_data_path = 'D:/Testing_seg_png'
sorts = ['FLAIR', 'T1', 'T1CE', 'T2']

for patient_folder in os.listdir(data_path):
    # Segmentation image load
    seg_images = []
    for seg_image in os.listdir(os.path.join(data_path,patient_folder,'SEG')):
        seg_images.append(cv2.imread(os.path.join(data_path,patient_folder, 'SEG', seg_image), 0))
    flair_images=[]
    flair_names=[]
    t1_images=[]
    t1_names=[]
    t1ce_images=[]
    t1ce_names=[]
    t2_images=[]
    t2_names=[]

    # each data load for patients
    for sort in sorts:
        for sort_image in os.listdir(os.path.join(data_path,patient_folder,sort)):
            if sort is 'FLAIR':
                flair_images.append(cv2.imread(os.path.join(data_path,patient_folder,sort,sort_image), 0))
                flair_names.append(sort_image)
            elif sort is 'T1':
                t1_images.append(cv2.imread(os.path.join(data_path,patient_folder,sort,sort_image), 0))
                t1_names.append(sort_image)
            elif sort is 'T1CE':
                t1ce_images.append(cv2.imread(os.path.join(data_path,patient_folder,sort,sort_image), 0))
                t1ce_names.append(sort_image)
            else:
                t2_images.append(cv2.imread(os.path.join(data_path,patient_folder,sort,sort_image), 0))
                t2_names.append(sort_image)

    # Make directory of each sortes and save the bitwise_and result
    os.makedirs(os.path.join(seg_data_path, patient_folder))
    for sort in sorts:
        os.makedirs(os.path.join(seg_data_path,patient_folder,sort))
        if sort is 'FLAIR':
            for i in range(len(flair_images)):
                img = cv2.bitwise_and(flair_images[i], seg_images[i])
                cv2.imwrite(os.path.join(seg_data_path,patient_folder,sort,flair_names[i]), img=img)

        if sort is 'T1':
            for i in range(len(t1_images)):
                img = cv2.bitwise_and(t1_images[i], seg_images[i])
                cv2.imwrite(os.path.join(seg_data_path,patient_folder,sort,t1_names[i]), img=img)

        if sort is 'T1CE':
            for i in range(len(t1ce_images)):
                img = cv2.bitwise_and(t1ce_images[i], seg_images[i])
                cv2.imwrite(os.path.join(seg_data_path,patient_folder,sort,t1ce_names[i]), img=img)

        if sort is 'T2':
            for i in range(len(t2_images)):
                img = cv2.bitwise_and(t2_images[i], seg_images[i])
                cv2.imwrite(os.path.join(seg_data_path,patient_folder,sort,t2_names[i]), img=img)

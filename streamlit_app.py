import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import shutil
import subprocess
from detect import detect
import sys
import argparse
import time
# -------------------------Input------------------------------#
# make a new folder save image
folder = os.path.join('images')
if not os.path.exists(folder):
    os.makedirs(folder)

folder_detect = os.path.join('yolov5/runs/detect')
if not os.path.exists(folder_detect):
    os.makedirs(folder_detect)

uploaded_file = st.sidebar.file_uploader(
    "Upload your image file ", type=['png', 'jpeg', 'jpg'])
if uploaded_file is not None:
    with st.spinner(text='loading...'):
        st.sidebar.image(uploaded_file)

        #resize image
        image = Image.open(uploaded_file)
        width, height = image.size
        new_width = int((width / height) * 640)
        image = image.resize((new_width, 640))
        image = image.save(f'images/{uploaded_file.name}')
        # int_image_path = f'images/{uploaded_file.name}'
else:
    st.error("Please upload a file")

number = st.sidebar.number_input('Enter the value of scale bar into the box below.')
st.sidebar.write('The current number is ', number)
 #-----------------------funtions-------------------------#
#  return all folder in path
def get_subdirs(b='.'):
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result
#  return newest folder
def get_detection_folder():
    return max(get_subdirs(os.path.join('yolov5/runs/detect')), key=os.path.getmtime)
    # return max(get_subdirs(os.path.join('detect')), key=os.path.getmtime)

# caculation diameter
diameter_core = []
diameter_shell = []
diameter_core1 = []
diameter_shell1 = []
def detect_diameter(namefile_txt, num_values):
    global D_core, D_shell, diameter_core, diameter_shell, diameter_core1, diameter_shell1
    # read file .txt
    annotation = pd.read_csv(namefile_txt, delimiter=' ', header=None,
                             names=['label', 'x_center', 'y_center', 'width', 'height'])
    label = annotation["label"]
    width = annotation['width']
    height = annotation['height']
    scale_bar = 1.0
    # calculate diameter core shell
    for i in range(0, len(annotation)):
        if (label[i] == 2):
            scale_bar = width[i]

    for i in range(0, len(annotation)):
        if (all(label[:] != 2)):
            break

        if (min(width[i], height[i]) / max(width[i], height[i]) >= 4/5):
            if (label[i] == 0):
                diameter_core = diameter_core + [((width[i] + height[i]) / (2*scale_bar))*num_values]
            if (label[i] == 1):
                diameter_shell = diameter_shell + [((width[i] + height[i]) / (2*scale_bar))*num_values]

        else:
        # if (min(width[i], height[i]) / max(width[i], height[i]) < 3 / 5):
            if (label[i] == 0):
                diameter_core = diameter_core + [(max(width[i], height[i]) / (scale_bar))*num_values]
            if (label[i] == 1):
                diameter_shell = diameter_shell + [(max(width[i], height[i]) / (scale_bar))*num_values]

    diameter_core = np.array(diameter_core)
    diameter_shell = np.array(diameter_shell)

    mean_shell = np.mean(diameter_shell)
    if (len(diameter_core) > 0):
        for i in range(0, len(diameter_core)):
            if len(diameter_shell)==0:
                diameter_core1 = diameter_core
                break
            # if diameter_core[i] < min(diameter_shell):
            if diameter_core[i] < mean_shell:
                diameter_core1 = diameter_core1 + [diameter_core[i]]
        D_core = np.mean(diameter_core1)
    else:
        D_core = np.NaN

    if (len(diameter_shell) > 0):
        for i in range(0, len(diameter_shell)):
            if np.isnan(D_core):
                diameter_shell1 = diameter_shell
                break
            if diameter_shell[i] > D_core:
                diameter_shell1 = diameter_shell1 + [diameter_shell[i]]
        D_shell = np.mean(diameter_shell1)
    else:
        D_shell = np.NaN  

    if not np.isnan(D_core) and not np.isnan(D_shell) :
        if ((len(diameter_core1)/len(diameter_shell1))<(2/10) ):
            D_core = np.NaN
        
 # ------------------------------# run detect.py in yolov5----------------------------------------
st.title('YOLOv5 Streamlit App')
if st.button("Run YOLOv5 Detection"):
    int_image_path = f'images/{uploaded_file.name}'
    path_detect_py = 'yolov5/detect.py'
    path_weight = "yolov5/runs/train/exp/weights/best.pt"
    # uot_path = 'yolov5/runs/detect'
   
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=path_weight , help='model path or triton URL')
        parser.add_argument('--source', type=str, default=int_image_path, help='file/dir/URL/glob/screen/0(webcam)')
        parser.add_argument('--conf-thres', type=float, default=0.55, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.1, help='NMS IoU threshold')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.set_defaults(save_txt=True)
        opt = parser.parse_args()
        # st.write(opt)
        detect(opt)
        time.sleep(7)
        
        for root, dirs, files in os.walk(get_detection_folder()):
            # st.write(root, dirs, files)
            for file in files:
                if (file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg')):
                    namefile_img = os.path.join(root, file)
                    st.image(namefile_img, caption='Image detected')
                    
                if file.endswith('.txt'):
                    namefile_txt = os.path.join(root, file)
                    detect_diameter(namefile_txt, number)
                    
        col1, col2 = st.columns([3, 1])
        if not np.isnan(D_core):
            with col1:
                # Tạo biểu đồ histogram core
                st.write("Diameter core is:", D_core)
                plt.figure(dpi=300)
                plt.hist(diameter_core1, bins=10, color='b', alpha=0.7)
                plt.ylabel('Frequency')
                plt.xlabel('Partical Diameter')
                plt.title('Histogram core')
                st.pyplot(plt)
            with col2:
                st.subheader("Diameter core")
                data = pd.DataFrame(({"Diameter_core": diameter_core1[:]}))
                st.dataframe(data, height=370, width=200)
        
        if not np.isnan(D_shell):
            with col1:
                # Tạo biểu đồ histogram shell
                st.write("Diameter shell is:", D_shell)
                plt.figure(dpi=300)
                plt.hist(diameter_shell1, bins=10, color='g', alpha=0.7)
                plt.ylabel('Frequency')
                plt.xlabel('Partical Diameter')
                plt.title('Histogram shell')
                st.pyplot(plt)
        
            with col2:
                st.subheader("Diameter shell")
                data = pd.DataFrame(({'Diameter_shell': diameter_shell1[:]}))
                st.dataframe(data, height=370, width=200)
        
        # Xóa tệp hình ảnh tạm thời
        time.sleep(7)
        # os.remove(int_image_path)
        shutil.rmtree('yolov5/runs/detect')
        shutil.rmtree('images')


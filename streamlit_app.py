import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
import subprocess
import time
# -------------------------Input------------------------------#
# make a new folder save image
folder = os.path.join('images')
if not os.path.exists(folder):
    os.makedirs(folder)

folder_detect = os.path.join('detect')
if not os.path.exists(folder_detect):
    os.makedirs(folder_detect)
    st.write(folder_detect)

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
    return max(get_subdirs(os.path.join('detect')), key=os.path.getmtime)

 # caculation diameter
diameter_core = []
diameter_shell = []
def detect_diameter(namefile_txt, num_values):
    global D_core, D_shell, diameter_core, diameter_shell
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

    if (len(diameter_shell) > 0):
        D_shell = np.mean(diameter_shell)
    else:
        D_shell = np.NaN

    if (len(diameter_core) > 0):
        D_core = np.mean(diameter_core)
    else:
        D_core = np.NaN
 # ------------------------------# run detect.py in yolov5----------------------------------------
st.title('YOLOv5 Streamlit App')
if st.button("Run YOLOv5 Detection"):
    int_image_path = f'images/{uploaded_file.name}'
    st.image(int_image_path)
    path_detect_py = 'yolov5/detect.py'
    iou = '0.1'
    out_path = 'detect'
    path_weight = "yolov5/runs/train/exp/weights/best.pt"
    command = ["python", path_detect_py,
               "--source", int_image_path,
               "--save-txt",
               "--weights", path_weight,
               "--iou-thres", iou,
               "--project", out_path]
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # st.write(process)
    time.sleep(7)

# if st.button("Run YOLOv5 Detection"):
    for root, dirs, files in os.walk(get_detection_folder()):
        path_detect = root
        for file in files:
            if file.endswith('.txt'):
                namefile_txt = os.path.join(root, file)
                detect_diameter(namefile_txt, number)
                st.write("Diameter core is:", D_core)
                st.write("Diameter core is:", D_shell)
            if (file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg')):
                namefile_img = os.path.join(root, file)
                st.image(namefile_img, caption='Image detected')
                
    col1, col2 = st.columns([3, 1])
    with col1:
        # Tạo biểu đồ histogram core
        plt.figure(dpi=300)
        plt.hist(diameter_core, bins=10, color='b', alpha=0.7)
        plt.ylabel('Frequency')
        plt.xlabel('Partical Diameter')
        plt.title('Histogram core')
        st.pyplot(plt)
        # Tạo biểu đồ histogram shell
        plt.figure(dpi=300)
        plt.hist(diameter_shell, bins=10, color='g', alpha=0.7)
        plt.ylabel('Frequency')
        plt.xlabel('Partical Diameter')
        plt.title('Histogram shell')
        st.pyplot(plt)

    with col2:
        st.subheader("Diameter core")
        data = pd.DataFrame(({"Diameter_core": diameter_core[:]}))
        st.dataframe(data, height=300, width=200)

        st.subheader("Diameter shell")
        data = pd.DataFrame(({'Diameter_shell': diameter_shell[:]}))
        st.dataframe(data, height=300, width=200)

    # Xóa tệp hình ảnh tạm thời
    os.remove(int_image_path)
    time.sleep(7)
    # shutil.rmtree('detect')



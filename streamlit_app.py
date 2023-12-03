import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import cv2
import os
import shutil
import subprocess
from detect import detect
import sys
import argparse
import time
from PyMieScatt import MieQCoreShell 
from scipy import interpolate
# -------------------------Input------------------------------#
# make a new folder save image
folder = os.path.join('images')
if not os.path.exists(folder):
    os.makedirs(folder)

folder_detect = os.path.join('yolov5/runs/detect')
if not os.path.exists(folder_detect):
    os.makedirs(folder_detect)

# uploaded_file = st.sidebar.file_uploader(
#     "Upload your image file ", type=['png', 'jpeg', 'jpg'])
uploaded_file = st.sidebar.file_uploader('', type=['png', 'jpeg', 'jpg'])
if uploaded_file is not None:
    with st.spinner(text='loading...'):
        st.sidebar.image(uploaded_file)

        #resize image
        image = Image.open(uploaded_file)
        image = image.convert('RGB')
        width, height = image.size
        new_width = int((width / height) * 640)
        image = image.resize((new_width, 640))

        
        # Điều chỉnh thay đổi độ sáng
        brightness_factor = np.random.uniform(0.7, 0.9)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

        red, green, blue = image.split() # change warmth
        red_enhancer = ImageEnhance.Brightness(red)
        red_adjusted = red_enhancer.enhance(1.15)
        image = Image.merge("RGB", (red_adjusted, green, blue))
        image.show(title='Brightened Image')
        image = image.save(f'images/{uploaded_file.name}')
else:
    st.sidebar.error("Please upload a file")

number = st.sidebar.number_input('Enter the value of scale bar into the box below.', step = 10)
# st.sidebar.write('The current number is ', number)

material_core = st.sidebar.selectbox(
    "Select material core",
    ['Au', 'Ag', 'Al2O3','Cu', 'Co', 'Cr', 'Fe2O3', 'Ge', 'MgO', 'Ni', 'Pb', 'Pt','Si', 'SiO2', 'TiO2'], index=None)
material_shell = st.sidebar.selectbox(
    "Select material shell",
    ['Au', 'Ag', 'Al2O3','Cu', 'Co', 'Cr', 'Fe2O3', 'Ge', 'MgO', 'Ni', 'Pb', 'Pt','Si', 'SiO2', 'TiO2'], index=None)

 #-----------------------funtions-------------------------#
#  ------------------return all folder in path-----------------------------
def get_subdirs(b='.'):
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result
#  ----------------------------------return newest folder------------------------------------
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
                             names=['label', 'x_center', 'y_center', 'width', 'height', 'conf'])
    label = annotation["label"]
    width = annotation['width']
    height = annotation['height']
    conf = annotation['conf']
    arr_scale_bar = []
    arr_conf = []
    # -------------------------------calculate diameter core shell---------------------------------------
    for i in range(0, len(annotation)):
        if (label[i] == 2):
            arr_scale_bar = arr_scale_bar + [width[i]]
            arr_conf = arr_conf + [conf[i]]
    if len(arr_scale_bar)>0:
        index_of_max_conf = arr_conf.index(max(arr_conf))
        scale_bar = arr_scale_bar[index_of_max_conf]

    for i in range(0, len(annotation)):
        if (all(label[:] != 2)):
            break

        if (min(width[i], height[i]) / max(width[i], height[i]) >= 4/5):
            if (label[i] == 0 and conf[i]>0.6):
                diameter_core = diameter_core + [((width[i] + height[i]) / (2*scale_bar))*num_values]
            if (label[i] == 1 and conf[i]>0.5):
                diameter_shell = diameter_shell + [((width[i] + height[i]) / (2*scale_bar))*num_values]

        else:
        # if (min(width[i], height[i]) / max(width[i], height[i]) < 3 / 5):
            if (label[i] == 0 and conf[i]>0.6):
                diameter_core = diameter_core + [(max(width[i], height[i]) / (scale_bar))*num_values]
            if (label[i] == 1 and conf[i]>0.5):
                diameter_shell = diameter_shell + [(max(width[i], height[i]) / (scale_bar))*num_values]

    diameter_core = np.array(diameter_core)
    diameter_shell = np.array(diameter_shell)

    mean_shell = np.mean(diameter_shell)
    mean_core = np.mean(diameter_core)
    
    if (len(diameter_core) > 0):
        for i in range(0, len(diameter_core)):
            if len(diameter_shell)==0:
                diameter_core1 = diameter_core
                break
            # if diameter_core[i] < min(diameter_shell):
            if diameter_core[i] < (mean_core+((mean_shell-mean_core)/2)):
                diameter_core1 = diameter_core1 + [diameter_core[i]]
        D_core = np.mean(diameter_core1)
    else:
        D_core = np.NaN

    if (len(diameter_shell) > 0):
        for i in range(0, len(diameter_shell)):
            if np.isnan(D_core):
                diameter_shell1 = diameter_shell
                break
            # if diameter_shell[i] > D_core and diameter_shell[i]<1.5*mean_shell:
            if diameter_shell[i]<1.5*mean_shell:
                diameter_shell1 = diameter_shell1 + [diameter_shell[i]]
        D_shell = np.mean(diameter_shell1)
    else:
        D_shell = np.NaN      

    if not np.isnan(D_core) and not np.isnan(D_shell) and len(diameter_shell1)>len(diameter_core1) :
        if ((len(diameter_core1)/len(diameter_shell1))<(0.4) ) or ((D_shell - D_core) < 0.05*num_values):
            D_core = np.NaN
            
    if not np.isnan(D_core) and not np.isnan(D_shell) and len(diameter_shell1)<len(diameter_core1) :
        if ((len(diameter_shell1)/len(diameter_core1))<(0.4) ) or ((D_shell - D_core) < 0.05*num_values):
            D_shell = np.NaN
#---------------------------interpolate dielectric function-----------------------------
def linear_interpolation(min_x, max_x, x_data, y_data, x_interpolate):
    if x_interpolate < min_x or x_interpolate > max_x:
        return 0 

    coefficients = interpolate.interp1d(x_data, y_data, kind='cubic')
    y_interpolate = coefficients(x_interpolate)# Tính giá trị nội suy
    return y_interpolate
 # ------------------------------# run detect.py in yolov5----------------------------------------
# st.title('YOLOv5 Streamlit App')
if st.button("Run Calculate"):
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
                plt.figure(dpi = 300)
                st.write("Core diameter is:", D_core)
                plt.figure(dpi=300)
                plt.hist(diameter_core1, bins=10, color='b', alpha=0.7)
                plt.ylabel('Frequency')
                plt.xlabel('Size')
                plt.title('Core diameter distribution')
                st.pyplot(plt)
            with col2:
                st.subheader("    ")
                data = pd.DataFrame(({"Diameter_core": diameter_core1[:]}))
                st.dataframe(data, height=420, width=200)
        
        if not np.isnan(D_shell):
            with col1:
                # Tạo biểu đồ histogram shell
                plt.figure(dpi = 300)
                st.write("Shell diameter is:", D_shell)
                plt.figure(dpi=300)
                plt.hist(diameter_shell1, bins=10, color='g', alpha=0.7)
                plt.ylabel('Frequency')
                plt.xlabel('Size')
                plt.title('Shell diameter distribution')
                st.pyplot(plt)
        
            with col2:
                st.subheader("    ")
                data = pd.DataFrame(({'Diameter_shell': diameter_shell1[:]}))
                st.dataframe(data, height=420, width=200)
            # -----------------Xóa tệp hình ảnh tạm thời---------------------------
        time.sleep(3)
        shutil.rmtree('yolov5/runs/detect')
        shutil.rmtree('images')
#------------------------------------------------------------------------------------------------------------------------------------------------------#
        if material_shell is not None and material_core is not None:
            if np.isnan(D_core) and np.isnan(D_shell):
                exit()
            # -------------------------Dielectric funtion core---------------------------------------------
            dielectric_core = pd.read_csv(f'Data_dielectric_function/{material_core}.csv', delimiter=',')
            lamda_core = dielectric_core['wl']
            lamda_core = (lamda_core.values)*1000
            
            n_core = dielectric_core['n']
            n_core = n_core.values
            
            column = 'k'
            if column in dielectric_core.columns:
                k_core = dielectric_core['k']
                k_core = k_core.values
            else:
                dielectric_core['k'] = 0
                k_core = dielectric_core['k']
                k_core = k_core.values
            # -------------------------Dielectric funtion shell-----------------------------
            dielectric_shell = pd.read_csv(f'Data_dielectric_function/{material_shell}.csv', delimiter=',')
            lamda_shell = dielectric_shell['wl']
            lamda_shell = (lamda_shell.values)*1000
            
            n_shell = dielectric_shell['n']
            n_shell = n_shell.values
            
            column = 'k'
            if column in dielectric_shell.columns:
                k_shell = dielectric_shell['k']
                k_shell = k_shell.values
            else:
                dielectric_shell['k'] = 0
                k_shell = dielectric_shell['k']
                k_shell = k_shell.values
            #-------------------------------#
            if np.isnan(D_core) and not np.isnan(D_shell): 
                D_core=0
                lamda_core = lamda_shell
                n_core = n_shell
                k_core = k_shell
            if not np.isnan(D_core) and np.isnan(D_shell): 
                D_shell=D_core
                lamda_shell = lamda_core
                n_shell = n_core
                k_shell = k_core
            min_x = max(lamda_core[0], lamda_shell[0])
            max_x = min(lamda_core[-1], lamda_shell[-1])
            if min_x <200:
                min_x = 200
            if max_x >3000:
                max_x = 3000
            wavelengths = np.linspace(min_x, max_x, 300)
            #-------------interpolate dielectric function  core with new wavelengths---------------------
            nCore = []
            kCore = []
            for x in wavelengths:
              result = linear_interpolation(min_x, max_x,lamda_core, n_core, x)
              nCore = nCore + [result]
            for x in wavelengths:
              result = linear_interpolation(min_x, max_x,lamda_core, k_core, x)
              kCore = kCore + [result]
            nCore = np.array(nCore)
            kCore = np.array(kCore)
            #-------------interpolate dielectric function shell with new wavelengths---------------------
            nShell = []
            kShell = []
            for x in wavelengths:
              result = linear_interpolation(min_x, max_x,lamda_shell, n_shell, x)
              nShell = nShell + [result]
            for x in wavelengths:
              result = linear_interpolation(min_x, max_x,lamda_shell, k_shell, x)
              kShell = kShell + [result]
            nShell = np.array(nShell)
            kShell = np.array(kShell)
            #------------------------------------------------------------------
            m_core = nCore + 1.0j * kCore
            m_shell = nShell + 1.0j * kShell
            # if np.isnan(D_core) and not np.isnan(D_shell): 
            #     D_core=0
            #     m_core = m_shell
            # if not np.isnan(D_core) and np.isnan(D_shell): 
            #     D_shell=D_core
            #     m_shell = m_core
            #---------------- caculate #qext, qsca, qabs, g, qpr, qback, qratio-----------------------------
            scattering_cross_sections = []
            for wavelength, mcore, mshell in zip(wavelengths, m_core, m_shell):
                    mie_core_shell = MieQCoreShell(wavelength=wavelength,dCore=D_core,dShell=D_shell,mCore=mcore,mShell=mshell)
                    scattering_cross_sections.append(mie_core_shell)
            
            scattering_cross_sections = np.array(scattering_cross_sections)
            # st.write(scattering_cross_sections)
            with col1:
                plt.figure(dpi = 300)
                fig, ax = plt.subplots()
                
                column_0 = scattering_cross_sections[:, 0]
                ax.plot( wavelengths,column_0,'b',  label='qext', marker='s', markersize=10, markevery=4)
                
                column_1 = scattering_cross_sections[:, 1]
                ax.plot( wavelengths,column_1,'r', label='qsca', marker='^', markersize=10, markevery=4)
                
                column_2 = scattering_cross_sections[:, 2]
                ax.plot( wavelengths,column_2,'g', label='qabs', marker='o', markersize=10, markevery=4)

                ax.set_xlim(200, 1000)
                ax.set_ylim(np.amin(scattering_cross_sections[:, 0:3]), np.amax(scattering_cross_sections[:, 0:3]))
                ax.set_title('The Effective Absorption, Scattering, and Extinction Spectra')
                ax.set_xlabel('Wavelength(nm)')
                ax.set_ylabel('Efficiency')
                ax.legend()
                st.pyplot(plt)
            with col2:
                st.subheader("")
                data = pd.DataFrame({'wl': np.linspace(min_x, max_x, 300),
                     'qext': scattering_cross_sections[:, 0],
                     'qsca': scattering_cross_sections[:, 1],
                     'qabs': scattering_cross_sections[:, 2]})
                st.dataframe(data, height=370, width=200)


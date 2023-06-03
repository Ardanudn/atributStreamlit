import streamlit as st
import torch
import subprocess
from detect import detect
from pathlib import Path
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import time
import cv2
from torchvision.utils import draw_bounding_boxes
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torchvision
import numpy as np

## CFG
cfg_model_path = "models/best.pt" 
confidence = .45
model = None

## END OF CFG

def imageInput(device, src):
    status = "Loading..."
    ada = "Loading..."
    tidak = "Loading..."

    if src == 'Upload your own data':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
                
        with st.columns(3)[1]:
            st.header("Status")
            st1_status = st.markdown("{status}".format(status=status))
        col1,col2,col3,col4 = st.columns(4)
        
        with col2:
            st2_ada = st.markdown("{ada}".format(ada=ada),unsafe_allow_html=True)
        with col3:
            st3_tidak = st.markdown("{tidak}".format(tidak=tidak),unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            img = img.resize((640, 640))
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            
            #call Model prediction--
            bbox, result,bbox_data = detect_image(image=imgpath,src="foto")
            imgp = create_bbox(img=imgpath,bbox=bbox,bbox_data=bbox_data,src="foto")

            with col2:
                status_img,atribut_ada,atribut_tidak_ada = count_atribut_image(result)
                st.image(imgp, caption='Model Prediction(s)', use_column_width='always')

                st1_status.markdown("{status}".format(status=status_img))
                st2_ada.markdown("{ada}".format(ada=atribut_ada),unsafe_allow_html=True)
                st3_tidak.markdown("{tidak}".format(tidak=atribut_tidak_ada),unsafe_allow_html=True)

    elif src == 'Sample data': 
        
        # Image selector slider
        imgpath = glob.glob('data/samples/images/*')
        imgsel = st.slider('Select random images from samples image.', min_value=1, max_value=len(imgpath), step=1)

        with st.columns(3)[1]:
            st.header("Status")
            st1_status = st.markdown("{status}".format(status=status))
        col1,col2,col3,col4 = st.columns(4)
        
        with col2:
            st2_ada = st.markdown("{ada}".format(ada=ada),unsafe_allow_html=True)
        with col3:
            st3_tidak = st.markdown("{tidak}".format(tidak=tidak),unsafe_allow_html=True)

        image_file = imgpath[imgsel-1]
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            img = img.resize((640, 640))
            st.image(img, caption='Selected Image', use_column_width='always')
        with col2:
            bbox, result,bbox_data = detect_image(image=image_file,size=(640,640),src="foto")
            img = create_bbox(img=image_file,bbox=bbox,bbox_data=bbox_data,src="foto")
            status_img,atribut_ada,atribut_tidak_ada = count_atribut_image(result)

            st1_status.markdown("{status}".format(status=status_img))
            st2_ada.markdown("{ada}".format(ada=atribut_ada),unsafe_allow_html=True)
            st3_tidak.markdown("{tidak}".format(tidak=atribut_tidak_ada),unsafe_allow_html=True)

            st.image(img, caption="Model prediction")



def videoInput(device, src,video=None):
    vid_file = None
    status = "Loading..."
    ada = "Loading..."
    tidak = "Loading..."
    
    if src == 'Sample data':
        vid_file = "data/samples/videos/{opt}.mp4".format(opt = video)
        outputpath = os.path.join('data/video_output', os.path.basename(vid_file))
    else:
        uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])
        if uploaded_video != None:

            ts = datetime.timestamp(datetime.now())
            vid_file = os.path.join('data/uploads', str(ts)+uploaded_video.name)
            outputpath = os.path.join('data/video_output', os.path.basename(vid_file))

            with open(vid_file, mode='wb') as f:
                f.write(uploaded_video.read())  # save video to disk

    with st.columns(3)[1]:
        st.header("Status")
        st1_status = st.markdown("{status}".format(status=status))

    col1,col2,col3,col4 = st.columns(4)
        
    with col2:
        st2_ada = st.markdown("{ada}".format(ada=ada),unsafe_allow_html=True)
    with col3:
        st3_tidak = st.markdown("{tidak}".format(tidak=tidak),unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    if vid_file:
      with col1:
        with st.spinner('Load Video'):
          video_file = open(vid_file, 'rb') #enter the filename with filepath

          video_bytes = video_file.read() #reading the file

          st.video(video_bytes)
      class_list_all_frames = []
      with col2:
        with st.spinner('Wait for it...'):
              cap = cv2.VideoCapture(vid_file)
              fps = cap.get(cv2.CAP_PROP_FPS)
              width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
              height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
              fourcc = cv2.VideoWriter_fourcc(*'mp4v')

              out = cv2.VideoWriter(outputpath, fourcc, fps, (width, height))
              while True:
                  ret, frame = cap.read()
                  if not ret:
                          break

                  gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                  model.conf = 0.45
                  result = model(gray_frame)
                  bbox_data = result.pandas().xyxy[0]
                  bbox = []
                  class_img = []
                  class_list_per_frame = []
                  for index, row in bbox_data.iterrows():
                    label = '{} {:.2f}'.format(row['name'], row['confidence'])
                    bbox.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                    class_img.append(label)
                    class_list_per_frame.append(row['class'])

                  class_list_all_frames.append(class_list_per_frame)

                  val_transform = A.Compose([
                            ToTensorV2()
                        ])
                  image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                  image = image/255
                  image = val_transform(image=image)
                  img_int = image['image'].clone().detach().mul(255).to(torch.uint8)
                  bbox = torch.tensor(bbox, dtype=torch.int)

                  if bbox.size() != torch.Size([0]):
                    img=draw_bounding_boxes(img_int, bbox, width=8)
                    image = torchvision.transforms.ToPILImage()(img)
                  else:
                    image= torchvision.transforms.ToPILImage()(img_int)

                  # Convert PIL image to OpenCV format
                  image_with_boxes = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                  for index, row in bbox_data.iterrows():
                        label = '{} {:.2f}'.format(row['name'], row['confidence'])
                        xmin, ymin, xmax, ymax = (
                            int(row['xmin']),
                            int(row['ymin']),
                            int(row['xmax']),
                            int(row['ymax']),
                        )
                        cv2.putText(
                            image_with_boxes,
                            label,
                            (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),  # font color (BGR format)
                            1,
                            cv2.LINE_AA,
                        )

                  # Write the frame with bounding boxes to the new video
                  out.write(image_with_boxes)


              cap.release()
              out.release()

        filename, extension = os.path.splitext(outputpath)
        output_file = os.path.join(filename+"codec" + extension)
        command = f"ffmpeg -i {outputpath} -vcodec libx264 {output_file}"

        try:
          subprocess.check_output(command, shell=True)
          print("Video conversion successful.")
        except subprocess.CalledProcessError as e:
          print("Video conversion failed:", e)
        video_file2 = open(output_file, 'rb') #enter the filename with filepath

        video_bytes2 = video_file2.read() #reading the file

        st.video(video_bytes2)


      st.success("Done")

      status_vid,atribut_ada,atribut_tidak_ada = count_atribut_video(class_list_all_frames)
      st1_status.markdown("{status}".format(status=status_vid))
      st2_ada.markdown("{ada}".format(ada=atribut_ada),unsafe_allow_html=True)
      st3_tidak.markdown("{tidak}".format(tidak=atribut_tidak_ada),unsafe_allow_html=True)


      with open(outputpath, 'rb') as file:
        st.sidebar.download_button(
              label="Download video result",
              data=file,
              file_name=os.path.basename(outputpath),
              mime="video/mp4")

def detect_image(image,src, size=None):
    model.conf = confidence
    if src == "foto":
        img = cv2.imread(image)
        reimg = cv2.resize(img, (640,640))
        gray_img = cv2.cvtColor(reimg, cv2.COLOR_RGB2GRAY)
        result = model(gray_img, size=size) if size else model(gray_img)
    else:
        result = model(image, size=size) if size else model(image)

    
    bbox_data = result.pandas().xyxy[0]
    bbox = []
    for index, row in bbox_data.iterrows():
        bbox.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])

    return bbox,result,bbox_data

def create_bbox(img,bbox,bbox_data,src):
  if src =="foto":
    val_transform = A.Compose([
                A.Resize(640, 640), # our input size can be 600px
                ToTensorV2()
            ])
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/255
    image = val_transform(image=image)
    img_int = image['image'].clone().detach().mul(255).to(torch.uint8)
  else:
    val_transform = A.Compose([
            ToTensorV2()
        ])
    
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = image/255
    image = val_transform(image=image)
    img_int = image['image'].clone().detach().mul(255).to(torch.uint8)
    

  bbox = torch.tensor(bbox, dtype=torch.int)
  if bbox.size() != torch.Size([0]):
    img=draw_bounding_boxes(img_int, bbox, width=8)
    image = torchvision.transforms.ToPILImage()(img)
  else:
    image= torchvision.transforms.ToPILImage()(img_int)

  if src =="foto":
    image_with_boxes = np.array(image)  
  else:
    image_with_boxes = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


  for index, row in bbox_data.iterrows():
        label = '{} {:.2f}'.format(row['name'], row['confidence'])
        xmin, ymin, xmax, ymax = (
            int(row['xmin']),
            int(row['ymin']),
            int(row['xmax']),
            int(row['ymax']),
        )
        cv2.putText(
            image_with_boxes,
            label,
            (xmin, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),  # font color (BGR format)
            1,
            cv2.LINE_AA,
        )

  return image_with_boxes

def count_atribut_image(result):

  for i in range(len(result.pandas().xyxy)):
    class_list = []
    for j in result.pandas().xyxy[i]['class']:
      class_list.append(j)
  
    if len(class_list) != 0:

      if set(class_list) == set([0, 1, 2, 3]):
        # lengkap = 1
        # frame_lengkap = i
        status = 'Atribut siswa lengkap!'
        break
      else:
        status = 'Atribut siswa tidak lengkap!'
    else:
      status = "Tidak ada atribut!"
      break

  atribut_ada = "Atribut yang dipakai:<br>"
  atribut_tidak_ada = "Atribut yang tidak dipakai:<br>"

  if len(class_list) != 0:
    if 0 in class_list:
      atribut_ada += "-- Badge<br>"
    else:
      atribut_tidak_ada += "<b>-- Badge</b<br>"
    if 1 in class_list:
      atribut_ada += "-- Dasi<br>"
    else:
      atribut_tidak_ada += "<b>-- Dasi</b><br>"
    if 2 in class_list:
      atribut_ada += "-- Sabuk<br>"
    else:
      atribut_tidak_ada += "<b>-- Sabuk</b><br>"
    if 3 in class_list:
      atribut_ada += "-- Topi<br>"
    else:
      atribut_tidak_ada += "<b>-- Topi</b><br>"
  
  if len(class_list) == 0:
    atribut_tidak_ada += "<b>-- Badge<br>-- Dasi<br>-- Sabuk<br>-- Topi</b>"
  
  # status += (atribut_ada + atribut_tidak_ada)

  return status,atribut_ada,atribut_tidak_ada

def count_atribut_video(class_list_all_frames):
  status = "Atribut siswa tidak lengkap!"
  minimum_deteksi_frame = 6
  count_0_badge = 0
  count_1_dasi = 0
  count_2_sabuk = 0
  count_3_topi = 0

  for i in class_list_all_frames:
    if len(i) != 0:
    #if len(result.pandas().xyxy[i]['class'].unique()) == 4:
      if set(i) == set([0,1,2,3]):
        count_0_badge += 1
        count_1_dasi += 1
        count_2_sabuk += 1
        count_3_topi += 1
        if ((count_0_badge >= minimum_deteksi_frame) and (count_1_dasi >= minimum_deteksi_frame) and (count_2_sabuk >= minimum_deteksi_frame) and (count_3_topi >= minimum_deteksi_frame)):
          status = "Atribut siswa lengkap!"
          break
        continue
      if 0 in i:
        count_0_badge += 1
      if 1 in i:
        count_1_dasi += 1
      if 2 in i:
        count_2_sabuk += 1
      if 3 in i:
        count_3_topi += 1
      if ((count_0_badge >= minimum_deteksi_frame) and (count_1_dasi >= minimum_deteksi_frame) and (count_2_sabuk >= minimum_deteksi_frame) and (count_3_topi >= minimum_deteksi_frame)):
        status = "Atribut siswa lengkap!"
        break
    else:
      continue

  atribut_ada = "Atribut yang dipakai:<br>"
  atribut_tidak_ada = "Atribut yang tidak dipakai:<br>"

  if status == "Atribut siswa lengkap!":
    # status += (atribut_ada + " --Badge<br>--Dasi<br>--Sabuk<br>--Topi" + atribut_tidak_ada)
    atribut_ada += (atribut_ada + "--Badge<br>--Dasi<br>--Sabuk<br>--Topi")
  else:
    if count_0_badge + count_1_dasi + count_2_sabuk + count_3_topi > 0:
      if count_0_badge >= minimum_deteksi_frame:
        atribut_ada += "--Badge<br>"
      else:
        atribut_tidak_ada += "--Badge<br>"
      if count_1_dasi >= minimum_deteksi_frame:
        atribut_ada += "--Dasi<br>"
      else:
        atribut_tidak_ada += "--Dasi<br>"
      if count_2_sabuk >= minimum_deteksi_frame:
        atribut_ada += "--Sabuk<br>"
      else:
        atribut_tidak_ada += "--Sabuk<br>"
      if count_3_topi >= minimum_deteksi_frame:
        atribut_ada += "--Topi<br>"
      else:
        atribut_tidak_ada += "--Topi<br>"
  
      # status += (atribut_ada + atribut_tidak_ada)
    else:
      # status += (atribut_ada + atribut_tidak_ada + "--Badge --Dasi --Sabuk --Topi")
      atribut_tidak_ada += (atribut_tidak_ada + "--Badge<br>--Dasi<br>--Sabuk<br>--Topi")

  return status,atribut_ada,atribut_tidak_ada

@st.cache_resource
def load_model(path, device):
    model_ = torch.hub.load('.', 'custom', path=path,source='local', force_reload=True)
    model_.to(device)
    print("model to ", device)
    return model_

def main():
    global model, confidence, cfg_model_path

    st.header('Atribut Detection')
    st.subheader('👈🏽 Select options left-haned menu bar.')

    # -- Sidebar
    st.sidebar.title('⚙️Options')
    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available!!!, please added to the model folder.", icon="⚠️")
    else:
        # device options
        if torch.cuda.is_available():
            deviceoption = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=1)
        else:
            deviceoption = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)
        
        # load model
        model = load_model(cfg_model_path, deviceoption)
        
        option = st.sidebar.radio("Select input type.", ['Image', 'Video'])

        datasrc = st.sidebar.radio("Select input source.", ['Sample data', 'Upload your own data'])
        

        # confidence slider
        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)

        # -- End of Sidebar

        if option == "Video" and datasrc == "Sample data":
          video = st.sidebar.selectbox(
            'Select Sample File',
            ('SD', 'SMP', 'SMA'))
          videoInput(deviceoption, datasrc,video)
        elif option == "Video" and datasrc == "Upload your own data": 
          videoInput(deviceoption, datasrc)
        elif option == "Image":    
          imageInput(deviceoption, datasrc)


if __name__ == '__main__':
  
    main()
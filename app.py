import streamlit as st
import torch
from detect import detect
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
    if src == 'Upload your own data':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        with st.columns(3)[1]:
                st.header("Status")
                st1_text = st.markdown("{status}".format(status=status))
                
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            
            #call Model prediction--
            bbox, result,bbox_data = detect_image(img=image_file,size=(640,640),src="foto")
            img = create_bbox(img=image_file,bbox=bbox,bbox_data=bbox_data,src="foto")
            img.save(outputpath)

            #--Display predicton
            
            img_ = Image.open(outputpath)
            with col2:
                status_img = count_atribut(result)
                st1_text.markdown("**{status}**".format(status=status_img))
                
                st.image(img_, caption='Model Prediction(s)', use_column_width='always')

    elif src == 'Sample data': 
        
        # Image selector slider
        imgpath = glob.glob('data/samples/images/*')
        imgsel = st.slider('Select random images from samples image.', min_value=1, max_value=len(imgpath), step=1)

        
        with st.columns(3)[1]:
            st.header("Status")
            st1_text = st.markdown("{status}".format(status=status))

        image_file = imgpath[imgsel-1]
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            img = img.resize((640, 640))
            st.image(img, caption='Selected Image', use_column_width='always')
        with col2:
            bbox, result,bbox_data = detect_image(img=image_file,size=(640,640),src="foto")
            img = create_bbox(img=image_file,bbox=bbox,bbox_data=bbox_data,src="foto")
            status_img = count_atribut(result)
            st1_text.markdown("**{status}**".format(status=status_img))
            #img = img.convert('RGB')
            st.image(img, caption="Model prediction")


def videoInput(device, src):
    vid_file = None
    if src == 'Sample data':
        vid_file = "data/samples/videos/sample.mp4"
        outputpath = os.path.join('data/video_output', os.path.basename(vid_file))
    else:
        uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])
        if uploaded_video != None:

            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+uploaded_video.name)
            outputpath = os.path.join('data/video_output', os.path.basename(imgpath))

            with open(imgpath, mode='wb') as f:
                f.write(uploaded_video.read())  # save video to disk

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

        out = cv2.VideoWriter(outputpath, fourcc, fps, (width, height))
        while True:
            ret, frame = cap.read()
            if not ret:
                    break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bbox, result, bbox_data = detect_image(img=gray_frame,size=(640,640),src="video")
            img = create_bbox(img=gray_frame,bbox=bbox,bbox_data=bbox_data,src="video")
            out.write(img)

        cap.release()
        out.release()


def detect_image(img,src, size=None):
    model.conf = confidence
    img = cv2.imread(img)
    #resize jadi ukuran 224x224
    reimg = cv2.resize(img, (640,640))
    # Convert menjadi gray
    if src == "foto":
        gray_img = cv2.cvtColor(reimg, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    result = model(gray_img, size=size) if size else model(gray_img)
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
  elif src == "video":
    val_transform = A.Compose([
            ToTensorV2()
        ])
    
  image = cv2.imread(img)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = image/255
  image = val_transform(image=image)
  img_int = image['image'].clone().detach().mul(255).to(torch.uint8)

  bbox = torch.tensor(bbox, dtype=torch.int)
  img=draw_bounding_boxes(img_int, bbox, width=8)
  image = torchvision.transforms.ToPILImage()(img)
  
  if src =="foto":
    image_with_boxes = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
  elif src == "video":
    image_with_boxes = image

    
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

def count_atribut(result):
  lengkap = 0
  frame_lengkap = -1

  for i in range(len(result.pandas().xyxy)):
    class_list = []
    for j in result.pandas().xyxy[i]['class']:
      class_list.append(j)
    if set(class_list) == set([0, 1, 2, 3]):
      lengkap = 1
      frame_lengkap = i
      status = 'Atribut siswa lengkap!'
      break
  if lengkap != 1:
    status = 'Atribut siswa tidak lengkap!'
  return status

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
            deviceoption = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
        else:
            deviceoption = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)
        
        # load model
        model = load_model(cfg_model_path, deviceoption)
        
        option = st.sidebar.radio("Select input type.", ['Image', 'Video'])

        datasrc = st.sidebar.radio("Select input source.", ['Sample data', 'Upload your own data'])

        # confidence slider
        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)

        # -- End of Sidebar

        
        if option == "Image":    
            imageInput(deviceoption, datasrc)
        elif option == "Video": 
            videoInput(deviceoption, datasrc)

if __name__ == '__main__':
  
    main()

from ultralytics import YOLO
from PIL import Image
import cv2
import os

#model = YOLO(r"D:\jupyter_code\ultralytics\runs\detect\train12\weights\best.pt")
model = YOLO("gf.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
#results = model.predict(source="0")
#results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

# from PIL
#img = Image.open(r"D:\jupyter_code\dataset\gf_data\1\DJI_20231129102242_0013_T.JPG")
#results = model.predict(source=img, save=True)  # save plotted images

# from ndarray
#img = cv2.imread(r"D:\jupyter_code\dataset\gf_data\1\DJI_20231129102242_0013_T.JPG")
#results = model.predict(source=img, save=True)  # save plotted imagesv
dir = "photo"
results = model.predict(source=dir, save=True, save_txt=True)  # save predictions as labels
print("len(results) = ",len(results))
# 使用yolov8进行预测
#results = model.predict(source=dir, mode="predict")#, conf=0.2)#, show=True)
ls = os.listdir(dir)
i = 0
for result in results:
    # Detection
    # annotator = Annotator(img)
    # print("result.obb:",result.obb)
    img = cv2.imread(dir+"\\"+ls[i])
    j = 0
    for rbox in result.obb:
        c = rbox.cls
        if c == 0:
            b = rbox.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            print("rbox: ",b)
            # Crop a license plate. Do some offsets to better fit a plate.
            rs_img = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
            cv2.imwrite(r'result\result_'+str(i)+'_'+str(j)+'.jpg', rs_img)
            j = j+1
    i = i+1
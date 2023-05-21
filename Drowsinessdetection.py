# Thư viện OpenCV để xử lý ảnh
import cv2
# Thư viện os để làm việc với các đường dẫn tập tin
import os
# Thư viện keras để load model
from keras.models import load_model
# Thư viện pygame để phát âm thanh
from pygame import mixer
import numpy as np
#import time
 
# Khởi tạo âm thanh
mixer.init()
sound = mixer.Sound('alarm.wav')

# Khởi tạo các bộ phân loại
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl=['Close','Open']

model = load_model('models/test1.h5')
path = os.getcwd()
# Mở camera
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    # Đọc một khung hình (frame) mới từ video hoặc camera và lưu trữ nó trong biến frame
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    # Tìm kiếm khuôn mặt,mắt trái, mắt phải trong khung hình bằng phương thức detectMultiScale 
    faces = face.detectMultiScale(frame,minNeighbors=5,scaleFactor=1.1,minSize=(161,161))
    left_eye = leye.detectMultiScale(frame)
    right_eye =  reye.detectMultiScale(frame)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED)

    # Tìm kiếm khuôn mặt trong khung hình và vẽ hình chữ nhật xung quanh khuôn mặt đó
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,0,0) ,1)

    # Tìm kiếm mắt trái trong khung hình và dự đoán mắt trái đó mở hay đóng
    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.resize(r_eye,(160,160))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(160,160,-3)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = np.argmax(model.predict(r_eye), axis=-1)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    # Tìm kiếm mắt phải trong khung hình và dự đoán mắt phải đó mở hay đóng
    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.resize(l_eye,(160,160))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(160,160,-3)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = np.argmax(model.predict(l_eye), axis=-1)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    # Nếu mắt trái và mắt phải đều đóng thì tăng biến score lên 1
    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>5):
        # Phát chuông cảnh báo
        sound.play()
        # Chụp ảnh và lưu vào thư mục images
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)   
        # Tăng giảm độ dày của frame   
        if(thicc<6):
            thicc=thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    # Nếu score nhỏ hơn 3 thì dừng phát âm thanh
    if(score<3):
        sound.stop()
    # Nếu nhấn phím q thì thoát chương trình
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

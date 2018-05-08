import numpy as np
import cv2
import string
import random
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
count=0
uname=''


def path(path):

	if not os.path.exists(path):
		os.makedirs(path)

def unique_name():

	global uname
	for i in range (0,15):
     		uname=uname+random.choice(string.ascii_letters + string.digits)
	return uname


def face_detect():

	while True:
		global uname,count,cap
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		font = cv2.FONT_HERSHEY_SIMPLEX
		for (x,y,w,h) in faces:
			count=count+1
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			cv2.putText(img, "face no. "+str(count), (x,y), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]
			roi_pic=roi_color
			 
			eyes = eye_cascade.detectMultiScale(roi_gray)
			for (ex,ey,ew,eh) in eyes:
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
		    		roi_pic = cv2.resize(roi_pic, (128,128)) #resizing images
		    	cv2.imwrite("/root/zoya/" + str(unique_name()) + '.jpg', roi_pic)
		uname=""
		count=0
		cv2.imshow('img',img)
		k = cv2.waitKey(30) & 0xff	
		if k == 27:
			break
	cap.release()
	cv2.destroyAllWindows()

path("/root/zoya/")
face_detect()

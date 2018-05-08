import numpy as np
import cv2
import string
import random
import os
import tensorflow as tf,sys
import time
from watchdog.observers import Observer  
from watchdog.events import PatternMatchingEventHandler  

face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
count=0
uname=''

class UpdateChecker(PatternMatchingEventHandler):	#containing functions to check for newly
	def process(self, event):			#created or modified files within a directory
		"""
		event.event_type 
        	'modified' | 'created' | 'moved' | 'deleted'
        	event.is_directory
        	True | False
        	event.src_path
            	path/to/observed/file
        	"""
        	# the file will be processed there
        	print event.src_path, event.event_type  # print now only for debug

    	def on_modified(self, event):
        	self.process(event)

    	def on_created(self, event):
        	self.process(event)


def update_checker(location):							#function checks the updates by calling the 											#functions of UpdateChecker Class
	observer = Observer()							
	observer.schedule(UpdateChecker(), path=location)
	observer.start()

	try:
		while True:
			time.sleep(10)
	except KeyboardInterrupt:
		observer.stop()

	observer.join()	


def path(path):					#checks if the path exists if not it creates one

	if not os.path.exists(path):
		os.makedirs(path)

def unique_name():				#provides a unique name by randomly picking 15 alphanumeric charecters

	global uname
	for i in range (0,15):
     		uname=uname+random.choice(string.ascii_letters + string.digits)
	return uname

def load_models():				#loading saved models from the database

	# Loads label file, strips off carriage return
	label_lines = [line.rstrip() for line in tf.gfile.GFile("/tf/output_labels.txt")]

	# Unpersists graph from file
	with tf.gfile.FastGFile("/tf/output_graph.pb", 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

def tf_face_recog():

	with tf.Session() as sess:    
		while True:
			
			image_path = "pics.jpeg"
			start=time.time()
			# Read in the image_data
			image_data = tf.gfile.FastGFile(image_path, 'rb').read()
			# Feed the image_data as input to the graph and get first prediction
			softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
			predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
			# Sort to show labels of first prediction in order of confidence
			top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
			for node_id in top_k:
				human_string = label_lines[node_id]
				score = predictions[0][node_id]
				net=(('%s (score = %.5f)' % (human_string, score)))
				#if score>0.7:
				print net
				print (time.time()-start)
				print "....................................................."



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

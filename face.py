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
img_name=''
net=''
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

def ModelNameArray():
	model_name_array=[]
	line_count=0
	#check the model_index.txt
	model_index=open("/root/zoya/model_index.txt")
	line= model_index.readlines()
	for model_name in line:
		line_count=line_count+1
	for model_name in line:
		model_name_array.append(model_name.replace("\n",""))

def load_models():				#loading saved models from the database

	new_model_name_array=[]
	line_count=0

	#check the model_index.txt
	model_index=open("/root/zoya/model_index.txt")
	line= model_index.readlines()
	for model_name in line:
		line_count=line_count+1
	for model_name in line:
		new_model_name_array.append(model_name.replace("\n",""))
	new_model_name_array=list(set(new_model_name_array)-set(ModelNameArray.model_name_array))
	for model in new_model_name_array:
		label_lines = [line.rstrip() for line in tf.gfile.GFile("/root/zoya/models/"+model+".txt")]

		# Unpersists graph from file
	     #with tf.gfile.FastGFile("/root/zoya/models/"+model+".pb", 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(tf.gfile.FastGFile("/root/zoya/models/"+model+".pb", 'rb').read())
		_ = tf.import_graph_def(graph_def, name='')
		
	# Loads label file, strips off carriage return
	label_lines = [line.rstrip() for line in tf.gfile.GFile("/tf/output_labels.txt")]

	# Unpersists graph from file
	with tf.gfile.FastGFile("/tf/output_graph.pb", 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')


with tf.Session() as sess:    
def tf_face_recog():
			
			global img_name,net
			image_path = "/root/zoya/" + str(img_name) + '.jpg'
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
				if score>0.7:
					print net
				print (time.time()-start)
				print "....................................................."




def face_detect():

	global uname,count,cap,img_name
	while True:

		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		font = cv2.FONT_HERSHEY_SIMPLEX
		for (x,y,w,h) in faces:
			count=count+1
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]
			roi_pic=roi_color
			 
			eyes = eye_cascade.detectMultiScale(roi_gray)
			for (ex,ey,ew,eh) in eyes:
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
		    		roi_pic = cv2.resize(roi_pic, (128,128)) #resizing images to 128x128 pixels
			img_name=unique_name()
		    	cv2.imwrite("/root/zoya/" + str(img_name) + '.jpg', roi_pic)
			cv2.putText(img, "ID: "+str(net), (x,y), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
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

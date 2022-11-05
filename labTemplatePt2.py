import numpy as np
import cv2
import time

import os

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import helper_scripts.annotator as annotator

partNum = 0


def detectBeesEnterExit(img, detector):
	rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# Create a TensorImage object from the RGB image.
	input_tensor = vision.TensorImage.create_from_array(rgb_image)

	# Run object detection estimation using the model.
	detection_result = detector.detect(input_tensor)
	
	for i in detection_result.detections:
		x = i.bounding_box.origin_x
		y = i.bounding_box.origin_y
		w = i.bounding_box.width
		h = i.bounding_box.height
		if 100 > h > 3 and 100 > w > 3:
			img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
	return img


def train():
	from tflite_model_maker.config import ExportFormat, QuantizationConfig
	from tflite_model_maker import model_spec
	from tflite_model_maker import object_detector
	from tflite_support import metadata
	import tensorflow as tf
	assert tf.__version__.startswith('2')

	tf.get_logger().setLevel('ERROR')
	from absl import logging
	logging.set_verbosity(logging.ERROR)

	train_data = object_detector.DataLoader.from_pascal_voc(
	    'bee_images/bee_validate',
	    'bee_images/bee_validate',
	    ['bee']
	)

	val_data = object_detector.DataLoader.from_pascal_voc(
	    'bee_images/bee_train',
	    'bee_images/bee_train',
	    ['bee']
	)
	spec = model_spec.get('efficientdet_lite0')
	model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=20, validation_data=val_data)
	model.evaluate(val_data)
	model.export(export_dir='.', tflite_filename='trained_models/bees.tflite')


def setUpDetector():
	base_options = core.BaseOptions(file_name="trained_models/bees.tflite", use_coral=False, num_threads=4)
	detection_options = processor.DetectionOptions(max_results=25, score_threshold=1)
	options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
	detector = vision.ObjectDetector.create_from_options(options)

	numRows = 1
	numColumns = 1
	return detector, numRows,numColumns


def detectBeesNormal(fullImage, detector, numRows, numColumns):
	locations = []

	j = 0
	while j<numRows:
		k = 0
		while k < numColumns:
			image = fullImage.copy()

			startX = int(image.shape[0]/numRows * j)
			endX = int(image.shape[0]/numRows * (j+1))
			startY = int(image.shape[1]/numColumns * k)
			endY = int(image.shape[1]/numColumns * (k+1))

			image = image[startX:endX, startY:endY]


			# Convert the image from BGR to RGB as required by the TFLite model.
			rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			# Create a TensorImage object from the RGB image.
			input_tensor = vision.TensorImage.create_from_array(rgb_image)

			# Run object detection estimation using the model.
			detection_result = detector.detect(input_tensor)

			for i in detection_result.detections:
				x = i.bounding_box.origin_x
				y = i.bounding_box.origin_y
				w = i.bounding_box.width
				h = i.bounding_box.height
				l = [i.bounding_box.origin_x+startX, i.bounding_box.origin_y+startY, i.bounding_box.width, i.bounding_box.height]
				locations += [l]
				color = (0,255,0)
				if w > 110 or h > 110:
					color = (0,0,255)
				else:
					cv2.rectangle(fullImage[startX:endX, startY:endY], (x,y), (x+w,y+h), color, 5)

			k+=1
		j+=1
	return fullImage


#________________________________________________________________________________________________________________
# dont edit any of the code below here:

def run():
	if partNum == 0:
		base_options = core.BaseOptions(
		file_name="trained_models/bee_enter_exit2.tflite", use_coral=False, num_threads=4)
		detection_options = processor.DetectionOptions(
		max_results=6, score_threshold=0.3)
		options = vision.ObjectDetectorOptions(
		base_options=base_options, detection_options=detection_options)
		detector = vision.ObjectDetector.create_from_options(options)

		cap = cv2.VideoCapture("images/realBees.avi")

		ret = True
		while True:
			(ret, img) = cap.read()
			if ret:	
				detected = detectBeesEnterExit(img.copy(), detector)

				cv2.imshow("d", detected)
				k = cv2.waitKey(1)
				if k == 27:
					break
			else:
				cap = cv2.VideoCapture("images/realBees.avi")
	elif partNum == 1:
		videosPath = "bee_images"
		saveName = ["bee"]
		modelName = ""
		annot = annotator.Annotator(videosPath, saveName, modelName)
		annot.goThroughDir()

	elif partNum == 2:
		train()

	elif partNum == 3:
		detector, r, c = setUpDetector()

		cap = cv2.VideoCapture("images/ex_video.mov")
		while True:
			(ret, img) = cap.read()
			if ret:	
				
				detected = detectBeesNormal(img, detector, r, c)
				detected = cv2.resize(detected, (1000,600))
				cv2.imshow("d", detected)
				k = cv2.waitKey(1)
				if k == 27:
					break
			else:
				cap = cv2.VideoCapture("images/ex_video.mov")

	else:
		print("no partNum", partNum)

run()
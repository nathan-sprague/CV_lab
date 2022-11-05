import numpy as np
import cv2
import time

import os

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import helper_scripts.annotator as annotator

partNum = 3


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
	    'queenVideo/bee_train',
	    'queenVideo/bee_train',
	    ['bee']
	)

	val_data = object_detector.DataLoader.from_pascal_voc(
	    'bee_images/bee_validate',
	    'bee_images/bee_validate',
	    ['bee']
	)
	spec = model_spec.get('efficientdet_lite0')
	model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=20, validation_data=val_data)
	model.evaluate(val_data)
	model.export(export_dir='.', tflite_filename='trained_models/queen.tflite')


def setUpDetector():
	base_options = core.BaseOptions(file_name="/home/nathan/Desktop/lab/trained_models/bee_nov4.tflite", use_coral=False, num_threads=4)
	detection_options = processor.DetectionOptions(max_results=25, score_threshold=0.1)
	options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
	detector = vision.ObjectDetector.create_from_options(options)

	numRows = 3
	numColumns = 3
	return detector, numRows,numColumns


def detectQueen(img, detector):
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

		if 500 > h > 3 and 500 > w > 3:
			img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 10)
	return img


#________________________________________________________________________________________________________________
# dont edit any of the code below here:

def run():

	if partNum == 0:
		videosPath = "queenVideo"
		saveName = ["queen"]
		modelName = ""
		annot = annotator.Annotator(videosPath, saveName, modelName)
		annot.goThroughDir()

	elif partNum == 1:
		train()

	elif partNum == 2:
		base_options = core.BaseOptions(
		file_name="trained_models/queen.tflite", use_coral=False, num_threads=4)
		detection_options = processor.DetectionOptions(
		max_results=3, score_threshold=0.2)
		options = vision.ObjectDetectorOptions(
		base_options=base_options, detection_options=detection_options)
		detector = vision.ObjectDetector.create_from_options(options)

		video="queenVideo/a_queen_IMG_2234.mov"
		cap = cv2.VideoCapture(video)
		cap.set(1,120)
		ret = True
		while True:
			(ret, img) = cap.read()

			if ret:	
				
				detected = detectQueen(img.copy(), detector)
				detected = cv2.resize(detected, (400,700))
				cv2.imshow("d", detected)
				k = cv2.waitKey(1)
				if k == 27:
					break
			else:
				cap = cv2.VideoCapture(video)
				cap.set(1,120)

	elif partNum == 3:
		base_options = core.BaseOptions(
		file_name="trained_models/queen.tflite", use_coral=False, num_threads=4)
		detection_options = processor.DetectionOptions(
		max_results=3, score_threshold=0.2)
		options = vision.ObjectDetectorOptions(
		base_options=base_options, detection_options=detection_options)
		detector = vision.ObjectDetector.create_from_options(options)

		video="images/a_queen_IMG_2189.mov"
		cap = cv2.VideoCapture(video)
		cap.set(1,200)
		ret = True
		while True:
			(ret, img) = cap.read()

			if ret:	
				
				detected = detectQueen(img.copy(), detector)
				detected = cv2.resize(detected, (700,400))
				cv2.imshow("d", detected)
				k = cv2.waitKey(1)
				if k == 27:
					break
			else:
				cap = cv2.VideoCapture(video)
				cap.set(1,200)

run()
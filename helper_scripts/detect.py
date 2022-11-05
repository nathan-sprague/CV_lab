# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time

import cv2

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
# import utils



def run():

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture("/home/nathan/Desktop/lab/images/ex_video.mov")
  
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name="/home/nathan/Desktop/lab/trained_models/bee_nov4.tflite", use_coral=False, num_threads=4)
  detection_options = processor.DetectionOptions(
      max_results=25, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  lastKey = 0
  while cap.isOpened():
    success, fullImage = cap.read()

    locations = []
    numRows = 2
    numColumns = 4
    # print(fullImage.shape)
    j = 0
    while j<numRows:
      k = 0
      while k < numColumns:
        image = fullImage.copy()

        startX = int(image.shape[0]/numRows * j)
        endX = int(image.shape[0]/numRows * (j+1))
        startY = int(image.shape[1]/numColumns * k)
        endY = int(image.shape[1]/numColumns * (k+1))
        # print(startX, endX, startY, endY)


        image = image[startX:endX, startY:endY]
     
        counter += 1

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
          #else:
          cv2.rectangle(fullImage[startX:endX, startY:endY], (x,y), (x+w,y+h), color, 5)
        # cv2.imshow(str(j) + str(k), image)
        # j=1000
        # break

        k+=1
      j+=1
    
    
    image = cv2.resize(fullImage, (1000,600))
    cv2.imshow('object_detector', image)

    # Stop the program if the ESC key is pressed.
    if lastKey == 32:
      lastKey = cv2.waitKey(0)
    else:
      lastKey = cv2.waitKey(1)

    if lastKey == 27:
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':

     run()
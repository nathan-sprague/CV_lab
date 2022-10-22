import os
import cv2
import json


imageDir = os.path.dirname(os.path.abspath(__file__))

path = imageDir + "/bee_images"
saveName = "bee"

modelName = "" # "/home/nathan/Desktop/ear_finder/ear_tflite/corn_ear_oct13.tflite" #"/home/nathan/Desktop/ear_finder/corn_ear_oct1.tflite"

if modelName != "":
	from tflite_support.task import core
	from tflite_support.task import processor
	from tflite_support.task import vision


class Annotator:
	def __init__(self, path, saveType, modelName=""):
		self.annotationName = "beeAnnotation"
		self.saveType = saveType
		self.mouseDown = False
		self.dragged = False
		self.openRects = []
		self.currentRectStart = [-1,-1]
		self.possibleRects = []
		self.path = path
		self.firstCallback = True
		self.allRects = {}
		dirs = os.listdir(path)
		if self.annotationName + ".json" in dirs:
			with open(path + '/' + self.annotationName + '.json') as json_file:
				data = json.load(json_file)
			print(data)
			self.allRects = data
		self.useDetector = False
		if modelName != "":
			self.useDetector = True
			base_options = core.BaseOptions(
			  file_name=modelName, use_coral=False, num_threads=4)
			detection_options = processor.DetectionOptions(
			  max_results=5, score_threshold=0.2)
			options = vision.ObjectDetectorOptions(
			  base_options=base_options, detection_options=detection_options)
			self.detector = vision.ObjectDetector.create_from_options(options)



	def clickAndMove(self, event, x, y, flags, param):
		# print(event, x, y, flags, param)

		if event==1:
			print("click")
			self.mouseDown = True
			self.openRects += [[x,y,x,y,0]]
			self.dragged = False

		elif event==4:
			print("release")
			self.mouseDown = False
			if abs(self.openRects[-1][0]-self.openRects[-1][2]) < 2 and abs(self.openRects[-1][1]-self.openRects[-1][3]) < 2:
				self.openRects = self.openRects[0:-1]


			if self.dragged == False:

				j=0
				while j<len(self.possibleRects):
					i = self.possibleRects[j]
					if x > min((i[0], i[2])) and x < max((i[0], i[2])):
						if y > min((i[1], i[3])) and y < max((i[1], i[3])):
							i+=[0]
							self.openRects += [i]
							self.possibleRects = self.possibleRects[0:j] + self.possibleRects[j+1::]
							j-=1
					j+=1

				j=0
				while j<len(self.openRects):
					i = self.openRects[j]
					self.openRects[j][4] = 0
					if x > min((i[0], i[2])) and x < max((i[0], i[2])):
						if y > min((i[1], i[3])) and y < max((i[1], i[3])):
							print("selected")
							self.openRects[j][4] = 1
					j+=1

				

				self.dispImg()


		elif self.mouseDown:
			self.dragged = True
			h, w, _ = self.ogImg.shape
			if x>w:
				x=w-1
			elif y>h:
				y=h-1
			if x<0:
				x=0
			elif y<0:
				y=0
			self.openRects[-1][2] = x
			self.openRects[-1][3] = y
			# print(x, y)
			self.dispImg()

			

	def dispImg(self):
		img = self.ogImg.copy()


		for i in self.openRects:
			if i[4] == 1:
				img = cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), (0,255,0), 2)
			elif i[4] == 0:
				img = cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), (255,0,0), 2)

		for i in self.possibleRects:
			img = cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), (0,255,255), 1)

			

		cv2.imshow("img", img)


		if self.firstCallback:
			cv2.setMouseCallback("img", self.clickAndMove)
			# cv2.createButton("img",None,cv2.QT_PUSH_BUTTON,1)
			self.firstCallback = False
			# cv2.createButton("img", self.buttonPressed, None,cv2.QT_PUSH_BUTTON,100)
			switch=0
			# cv2.createTrackbar("yo", 'img',0,1, self.buttonPressed)


	# def buttonPressed(self, val):
	# 	print("yo", val)


	def detectPossible(self):
		rgb_image = cv2.cvtColor(self.ogImg, cv2.COLOR_BGR2RGB)

		# Create a TensorImage object from the RGB image.
		input_tensor = vision.TensorImage.create_from_array(rgb_image)

		# Run object detection estimation using the model.
		detection_result = self.detector.detect(input_tensor)
		
		for i in detection_result.detections:
			x = i.bounding_box.origin_x
			y = i.bounding_box.origin_y
			w = i.bounding_box.width
			h = i.bounding_box.height
			if h > 3 and w > 3:
				self.possibleRects += [[x,y, x+w,y+h]]
		

	def goThroughFiles(self):
		allDirs = os.listdir(path)
		dirs = []
		for i in allDirs:
			if i[-4::] == ".avi" or i[-4::] == ".jpg" or i[-4::] == ".mov":
				dirs += [i]

		i=0

		while i < len(dirs):
			if i < 0:
				i=0
			iStart = i
			
			file = dirs[i]
			print("reading", file)

			
			self.openRects = []
			

			if file[-4::] == ".jpg":
				analyzedBefore = False
				if file in self.allRects:
					self.openRects = self.allRects[file]
					analyzedBefore = True
				
				self.ogImg = cv2.imread(path + "/" + file)

				self.possibleRects = []
				if self.useDetector and not analyzedBefore:
					self.detectPossible()

				self.dispImg()

				k = cv2.waitKey(0)
			
				if k == 27:
					break
				elif k == 8: # delete
					l=0
					deleted = 0
					while l<len(self.openRects):
						if self.openRects[l][4] == 1:
							self.openRects = self.openRects[0:l] + self.openRects[l+1::]
							deleted += 1
							l-=1
						l+=1
					if deleted == 0 and len(self.openRects) > 0:
						self.openRects = self.openRects[0:-1]

				elif k == 115:
					self.allRects[file] = self.openRects[:]
					self.saveDataJSON()
				elif k == 112: # save as xml
						self.saveDataXML()

				elif k == 81 or k == 106: # left
					print("past")
					i-=1
				else:
					if i < len(dirs)-1:
						i+=1
					else:
						print("reached last image")

				if self.openRects[:] != []:
					self.allRects[file] = self.openRects[:]
					self.openRects = []

			elif file[-4::] == ".avi" or file[-4::] == ".mov":
				cap = cv2.VideoCapture(path + "/" + file)
				ret = True
				k = 0
				j=0
				jStart = 0
				frameChange = True
				if file not in self.allRects:
					self.allRects[file] = {}

				while cap.isOpened() and ret:

					analyzedBefore = False
					if str(j) in self.allRects[file]:
						self.openRects = self.allRects[file][str(j)]
						analyzedBefore = True
					jStart = j
					# print("frame?", cap.get(cv2.CAP_PROP_POS_FRAMES))
					if frameChange:
						ret, self.ogImg = cap.read()
						print("reading frame",j)
					if not ret:
						print("done reading this image")
						break
					frameChange = False

					self.possibleRects = []
					if self.useDetector and not analyzedBefore:
						self.detectPossible()
					
					self.dispImg()

					k = cv2.waitKey(0)
					print(k)


					if k == 27:
						i=len(dirs)
						break
				
					elif k == 81 or k == 106: # left
						print("back 1 frame (watch out ineffient)")

						cap = cv2.VideoCapture(path + "/" + file)
						m = 0
						if j == 0:
							i-=2
							break

						while m < j-1: # watch out: this is pretty inefficient. Use sparingly
							# print(m)
							cap.read()
							m += 1
						j-=1
						frameChange = True


						if False: # opencv doesn't let you go backwards (right now)
							j-=1
							print("left")

							next_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
							current_frame = next_frame - 1
							previous_frame = current_frame - 1
							if previous_frame >= 0:
	  							cap.set(cv2.CAP_PROP_POS_FRAMES, previous_frame)
	  							print("cap set")

							if j < 0:
								i-=2
								break

					elif k == 112: # save as xml
						self.saveDataXML()

					elif k == 115:
						self.allRects[file][str(j)] = self.openRects[:]
						self.saveDataJSON()

					elif k == 108:
						print("skip 5")
						m = 0
						j+=1
						for m in range(0,5):
							cap.read()
							j+=1
						frameChange = True

					elif k == 8: # delete
						l=0
						deleted = 0
						while l<len(self.openRects):
							if self.openRects[l][4] == 1:
								self.openRects = self.openRects[0:l] + self.openRects[l+1::]
								l-=1
								deleted += 1
							l+=1
						if deleted == 0 and len(self.openRects) > 0:
							self.openRects = self.openRects[0:-1]

					elif k == 91: # '[' - go to previously labeled image
						closestSmall = -1
						keys = list(self.allRects[file])
						keys.sort(key=int)

						m=0
						while m < len(keys):
							check = keys[m]
							if int(check) >= j:
								break
							elif self.allRects[file][check] != []:
								closestSmall = int(check)
							m += 1
						print("closest small", closestSmall)
						if closestSmall != -1:
							j=0
							cap = cv2.VideoCapture(path + "/" + file)
							while j < closestSmall:
								cap.read()
								j+=1
							frameChange = True

					elif k == 93: # ']' - go to next labeled image
						closestBig = -1
						keys = list(self.allRects[file])
						keys.sort(key=int, reverse=True)

						m=0
						while m < len(keys):
							check = keys[m]
							if int(check) <= j:
								break
							elif self.allRects[file][check] != []:
								closestBig = int(check)
							m += 1
						print("closest big", closestBig)
						if closestBig != -1:
							while j < closestBig-1:
								cap.read()
								j+=1
							j+=1
							frameChange = True

					elif k == 92: # '\' - go to end of labeled images (in video)
						closestBig = -1
						keys = list(self.allRects[file])
						keys.sort(key=int, reverse=True)

						print("going to last labeled image... please wait")
						if len(keys) > 0:
							if int(keys[0]) > j:
								while j < int(keys[0])-1:
									cap.read()
									j+=1
								j+=1
								frameChange = True
						print("got it.")


					elif k == 47: # '/' - go to next image/video
						break


					else:
						frameChange = True
						j+=1

					if self.openRects[:] != []:
						if self.mouseDown == False:
							m = 0
							while m<len(self.openRects):
								orr = self.openRects[m]
								if abs(orr[0] - orr[2]) < 3 or abs(orr[1] - orr[3]) < 3:
									self.openRects = self.openRects[0:m] + self.openRects[m+1::]
									m-=1
									print("removed tiny rectangle")
								m+=1
						self.allRects[file][str(jStart)] = self.openRects[:]
						self.openRects = []

				if self.openRects[:] != []:
					self.allRects[file][str(jStart)] = self.openRects[:]
					self.openRects = []

				i+=1

			else: 
				print("Error: should never get here")
				exit()

		if i == len(dirs):
			print("finished labeling all the images. Saving.")
			self.saveDataJSON()


	def reduceJSON(self):
		k = list(self.allRects.keys())
		reduced = {}
		numEars = 0
		numImgs = 0
		for i in k:
			if i[-4::] == ".jpg":
				if self.allRects[i] != []:
					reduced[i] = self.allRects[i]
					numImgs += 1
					numEars += len(self.allRects[i])
			else:
				l = list(self.allRects[i].keys())
				dictToAdd = {}
				for j in l:
					if self.allRects[i][j] != []:
						dictToAdd[j] = self.allRects[i][j]

						numEars += len(self.allRects[i][j])
						numImgs += 1

				if len(list(dictToAdd.keys())) > 0:
					reduced[i] = dictToAdd 
		print("reduced", reduced)
		print("recorded", numEars, "ears from", numImgs, "images")
		return reduced


	def saveDataJSON(self):
		reduced = self.reduceJSON()
		
		
		jsonStr = json.dumps(reduced)
		with open(self.path + "/" + self.annotationName + ".json", 'w') as fileHandle:
			fileHandle.write(str(jsonStr))
			fileHandle.close()


	def saveDataXML(self):
		print("saving xml files")
		trainDirName = self.path + "/train"
		if not os.path.exists(trainDirName):
			os.makedirs(trainDirName)
		validDirName = self.path + "/validate"
		if not os.path.exists(validDirName):
			os.makedirs(validDirName)

		reduced = self.reduceJSON()

		imgCount = 0
		for i in reduced:
			if i[-4::] == ".jpg":
				imgCount+=1
				dirName = trainDirName
				if imgCount%10 == 0:
					dirName = validDirName
				img = cv2.imread(self.path + "/" + i)
				h, w, _ = img.shape
				self.saveXML(dirName + "/", i, reduced[i], [w,h], i)
				cv2.imwrite(dirName + "/" + i, img)
				
			if i[-4::] == ".avi" or i[-4::] == ".mov":
				cap = cv2.VideoCapture(self.path + "/" + i)
				ret = True
				keys = list(reduced[i].keys())
				keys.sort(key = int)
				print("saving annotations in video", i)
				print("sorted keys", keys)

				k=-1
				for j in keys:
					
					while k<int(j):
						ret, img = cap.read()
						# cv2.waitKey(1)
						k+=1
						# print(k)
					img2 = img.copy()
					for d in reduced[i][j]:
						img2 = cv2.rectangle(img2, (d[0], d[1]), (d[2], d[3]), (0,255,0), 2)
					# cv2.imshow("saving", img2)

					imgCount+=1
					dirName = trainDirName
					if imgCount%10 == 0:
						dirName = validDirName
					
					# cv2.waitKey(0)
					h, w, _ = img.shape
					saveName = "f" + j + "_" + i[0:-4] + ".jpg"
					self.saveXML(dirName + "/", saveName, reduced[i][j], [w,h], saveName)
					cv2.imwrite(dirName + "/" + saveName, img)
		print("Saved all")




	def saveXML(self, path, filename, objects, imshape, name):

		text = """<annotation>
		<folder>annotated2</folder>
		<filename>""" + filename + """</filename>
		<path>""" + os.path.abspath(path) + "/"  + filename + """</path>
		<source>
		<database>Unknown</database>
		</source>
		<size>
		<width>""" + str(imshape[0]) + """</width>
		<height>""" + str(imshape[1]) + """</height>
		<depth>3</depth>
		</size>
		<segmented>0</segmented>"""

		for i in objects:
			text += """<object>
			<name>""" + self.saveType + """</name>
			<pose>Unspecified</pose>
			<truncated>0</truncated>
			<difficult>0</difficult>
			<bndbox>
			<xmin>""" + str(min([i[0], i[2]])) + """</xmin>
			<ymin>""" + str(min([i[1], i[3]])) + """</ymin>
			<xmax>""" + str(max([i[0], i[2]])) + """</xmax>
			<ymax>""" + str(max([i[1], i[3]])) + """</ymax>
			</bndbox>
			</object>"""

		text += """\n</annotation>"""


		filename = filename[0:-4] + ".xml"
		print("saving data as", filename)
		with open(path + filename, 'w') as fileHandle:
			fileHandle.write(text)
			fileHandle.close()

ann = Annotator(path, saveName, modelName)
ann.goThroughFiles()
import numpy as np
import cv2
import time
import labTools
import os


partNum = 0


def showImage(img):
	cv2.imshow("image", img)
	cv2.waitKey(0)

def makeGrayscalePython(img):
	h, w = img.shape[:2]
	gray = np.zeros((h,w,1), dtype=np.uint8) # make empty image
	start = time.time()
	for i in range(0, w): # loop through every pixel in width
		for j in range(0, h): # loop through every pixel in height
			b = int(img[j][i][0]) 
			g = int(img[j][i][1])
			r = int(img[j][i][2])
			gray[j][i][0] = int((b+g+r)/3) # calculate average here
	end = time.time()
	print("python alone took", end-start, "seconds")
	return gray


def makeGrayscaleNumpy(img):
	start = time.time()
	gray = img[:,:,0] # get the red pixel channel
	end = time.time()
	print("numpy took", end-start, "seconds")
	return gray

def makeGrayscaleOpenCV(img):
	start = time.time()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	end = time.time()
	print("opencv took", end-start, "seconds")
	return gray


def bgrThreshold(img):
	lowerThresh = np.array([25,25,80]) # change threshold values 
	upperThresh = np.array([120,120,255]) # change threshold values 

	mask = cv2.inRange(img, lowerThresh, upperThresh)
	res = cv2.bitwise_and(img, img, mask=mask) # make everything outside the threshold black

	return res


def hsvThreshold(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # convert image to HSV
	cv2.imshow("hsv", hsv)

	lowerThresh = np.array([0,0,0]) # change threshold values
	upperThresh = np.array([95,255,255]) # change threshold values
	mask = cv2.inRange(hsv, lowerThresh, upperThresh)
	res = cv2.bitwise_and(img, img, mask= mask) # make everything outside the threshold black

	return res


def detectCanny(img):
	thresh1 = 150 # change threshold
	thresh2 = 300 # change threshold
	canny = cv2.Canny(img, thresh1, thresh2) # create canny lines

	kernel = np.ones((10, 10))
	dilated = cv2.dilate(canny, kernel, 1) # make lines wider

	img[dilated < 20] = 0 # remove areas without the canny lines from the image

	return img


def blobDetector(img):
	thresh = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	thresh[thresh>0] = 255 # make image black and white
	cv2.imshow("thres", thresh)

	contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # get contours

	minArea = 500 # change
	maxArea = 10000 # change 

	beeLocations = []
	for con in contours: # look at every contour found
		area = cv2.contourArea(con)
		x,y,w,h = cv2.boundingRect(con)
		if area > minArea and area < maxArea: # only add coordinates to locations list if it is the right size
			beeLocations += [cv2.boundingRect(con)]

	return beeLocations

	
def detectBees1(img):

	canny = detectCanny(img.copy()) # from earlier function

	beeLocations = blobDetector(canny)

	for i in beeLocations:
		x,y,w,h = i
		img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) # make rectangle around bee

	return img


def detectBees2(img):
	canny = detectCanny(img.copy())

	beeLocations = blobDetector(canny)

	minPercentColor = 0.8 # change

	for i in beeLocations:
		x,y,w,h = i
		print(i)
		bee = hsvThreshold(img[y:y+h,x:x+w])
		nz = np.count_nonzero(bee)
		percentColor = nz/(h*w*3) # get percent of pixels that fall within the color threshold

		if percentColor > minPercentColor: # filter by color
			img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

	return img


def detectBees3(img, lastImg):
	canny = detectCanny(img.copy())

	beeLocations = blobDetector(canny)

	change = cv2.absdiff(src1=lastImg, src2=img) # get difference between the two images


	kernel = np.ones((5, 5))
	dilated = cv2.dilate(change, kernel, 1) # dilate the change to exagerate differences in movement
	
	change = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
	

	minPercentColor = 0.7 # change
	minPercentMove = 0.5 # change

	for i in beeLocations:
		x,y,w,h = i

		bee = hsvThreshold(img[y:y+h, x:x+w])
		nz = np.count_nonzero(bee)
		percentColor = nz/(h*w*3) # get percent of area within the color threshold

		changedArea = dilated[y:y+h,x:x+w]
		nz = np.count_nonzero(changedArea)
		percentMoved = nz/(h*w)  # get percent of area that moved

		if percentColor > minPercentColor and percentMoved > minPercentMove: # the blob meets all the criteria
			img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

	return img




#________________________________________________________________________________________________________________
# dont edit any of the code below here:


def run():



	imageDir = os.path.dirname(os.path.abspath(__file__))


	dir = imageDir + "/images/wood.jpg"
	wood = cv2.imread(dir)

	dir = imageDir + "/images/bee1.jpg"
	beeImg = cv2.imread(dir)

	dir = imageDir + "/images/spider.jpg"
	spiderImg = cv2.imread(dir)

	dir = imageDir + "/images/knot.jpg"
	knotImg = cv2.imread(dir)

	if partNum == 0:
		showImage(beeImg)

	elif partNum == 1:
		res = makeGrayscalePython(beeImg)
		cv2.imshow("Part 1", res)

	elif partNum == 2:
		res = makeGrayscaleNumpy(beeImg)
		cv2.imshow("Part 2", res)

	elif partNum == 3:
		res = makeGrayscaleOpenCV(beeImg)
		cv2.imshow("Part 3", res)

	elif partNum == 4:
		res = bgrThreshold(beeImg)
		cv2.imshow("part 4", res)

	elif partNum == 5:
		res = hsvThreshold(beeImg)
		cv2.imshow("Part 5", res)

	elif partNum == 6:
		img = labTools.generateRandomBees(wood.copy(), beeImg.copy(), numBees=3)
		res = hsvThreshold(img)
		cv2.imshow("Part 6", res)

	elif partNum == 7:
		img = labTools.generateRandomBees(wood.copy(), beeImg.copy(), numBees=3)
		res = detectCanny(img)
		cv2.imshow("Part 7", res)

	elif partNum == 8:
		img = labTools.generateRandomBees(wood.copy(), beeImg.copy(), numBees=3)
		res = detectBees1(img)
		cv2.imshow("Part 8", res)

	elif partNum == 9:
		img = labTools.generateRandomBees(wood.copy(), beeImg.copy(), spiderImg=spiderImg, numBees=4)
		res = detectBees1(img)
		cv2.imshow("Part 9", res)

	elif partNum == 10:
		img = labTools.generateRandomBees(wood.copy(), beeImg.copy(), spiderImg=spiderImg, numBees=4)
		res = detectBees2(img)
		cv2.imshow("Part 10", res)

	elif partNum == 11:
		img = labTools.generateRandomBees(wood.copy(), beeImg.copy(), spiderImg=spiderImg, knotImg=knotImg, numBees=4)

		res = detectBees2(img)
		cv2.imshow("Part 11", res)

	elif partNum == 12:
		lastRes, lastLocations = labTools.generateRandomBees(wood.copy(), beeImg.copy(), numBees=5, spiderImg=spiderImg, knotImg=knotImg, lastLocations=True)
		while True:
			res, lastLocations = labTools.generateRandomBees(wood.copy(), beeImg.copy(), numBees=5, spiderImg=spiderImg, knotImg=knotImg, lastLocations=lastLocations)
			detected = detectBees3(res.copy(), lastRes.copy())
			cv2.imshow("Part 12", detected)
			lastRes = res
			time.sleep(0.1)
			k = cv2.waitKey(1)
			if k == 27:
				break

	elif partNum == 13:
		cap = cv2.VideoCapture(imageDir + "/images/realBees.avi")
		ret = True
		(ret, oldImg) = cap.read()
		while ret:
			(ret, img) = cap.read()
			if ret:	
				detected = detectBees3(img.copy(), oldImg.copy())
				cv2.imshow("d", detected)
				oldImg = img
				k = cv2.waitKey(0)
				if k == 27:
					break
	else:
		print("there is no part", partNum)
	if partNum < 12:
		cv2.waitKey(0)


run()



import numpy as np
import cv2

def generateRandomBees(hive, beeImg, numBees=1, spiderImg=None, knotImg=None, lastLocations=None, cascades=False):
	locations = []
	hiveH, hiveW, _ = hive.shape

	for i in range(0,numBees):


		if lastLocations is not None and lastLocations is not True:
			scale = lastLocations[i][3]
			brightness = lastLocations[i][4]
			angle = lastLocations[i][5]
		elif cascades:
			angle = np.random.rand()*50+40
			if angle > 80:
				angle = np.random.rand()*50
			brightness = int(np.random.rand() * 100 - 50)
			scale = np.random.rand() * 0.21
			if scale < 0.18:
				scale += 0.18
		else:
			angle = np.random.rand()*180
			brightness = int(np.random.rand() * 100 - 50)
			scale = np.random.rand() * 0.25
			if scale < 0.2:
				scale += 0.2


		useSpider = False
		useKnot = False
		if i == 0 and spiderImg is not None:
			useSpider = True
			beeDrawn = spiderImg.copy()
		elif i == 1 and knotImg is not None:
			useKnot = True
			scale = 0.3
			beeDrawn = knotImg.copy()

		else:
			beeDrawn = beeImg.copy()


		h, w, _ = beeDrawn.shape
		beeDrawn = cv2.resize(beeDrawn, (int(w*scale), int(h*scale) ) , interpolation = cv2.INTER_AREA)
		h, w, _ = beeDrawn.shape

		if not useSpider and not useKnot:
			center = (w/2, h/2)
			
			rotate_matrix = cv2.getRotationMatrix2D(center=center, angle= angle, scale=1)
			beeDrawn = cv2.warpAffine(src=beeDrawn, M=rotate_matrix, dsize=(h, w))

		h, w, _ = beeDrawn.shape
		hsv = cv2.cvtColor(beeDrawn.copy(), cv2.COLOR_BGR2HSV)
		if useSpider:
			hsv = beeDrawn.copy()
			lower_red = np.array([0,0,0])
			upper_red = np.array([200,200,200])
			mask = cv2.inRange(hsv, lower_red, upper_red)

			beeDrawn = cv2.bitwise_and(beeDrawn, beeDrawn, mask= mask)
		elif useKnot:
			hsv = beeDrawn.copy()
			lower_red = np.array([0,0,0])
			upper_red = np.array([230,230,230])
			mask = cv2.inRange(hsv, lower_red, upper_red)
			beeDrawn = cv2.bitwise_and(beeDrawn, beeDrawn, mask= mask)

		else:
			lower_red = np.array([1,0,0])
			upper_red = np.array([95,255,255])
			mask = cv2.inRange(hsv, lower_red, upper_red)
			
			beeDrawn = cv2.bitwise_and(beeDrawn, beeDrawn, mask= mask)
	
		if not useSpider and not useKnot:
			
			if brightness > 0:
				beeDrawn[beeDrawn<255-brightness] += brightness
				beeDrawn[beeDrawn==brightness] = 0
			else:
				beeDrawn[beeDrawn>-brightness] -= abs(brightness)

		
		if lastLocations == None or lastLocations == True:
			acceptableLocation = False
			iterations = 0
			while not acceptableLocation and iterations < 10:
				acceptableLocation = True
				speed = 10
				speeds = [int(np.random.rand()*speed)-int(speed/2), int(np.random.rand()*speed)-int(speed/2)]
				if abs(speeds[0])<1:
					speeds[0]=1
				elif abs(speeds[1])<1:
					speeds[1]=1
				location = [max([0, int(np.random.rand()*hiveW-w)]), max([0,int(np.random.rand()*hiveH-h)]), speeds, scale, brightness, angle]
				for i in locations:
					if abs(i[0]-location[0]) < 80 and abs(i[1]-location[1]) < 80:
						acceptableLocation = False
						break
				iterations += 1
			if iterations == 10:
				continue
		else:
			if i != 1:
				location = [lastLocations[i][0]+lastLocations[i][2][0], lastLocations[i][1]+lastLocations[i][2][1],  lastLocations[i][2], lastLocations[i][3], lastLocations[i][4],  lastLocations[i][5]]
			else:
				location = lastLocations[i]


		if location[0]+w > hiveW:
			location[0] = 0
		elif location[0] < 0:
			location[0] = hiveW-w

		if location[1]+h > hiveH:
			location[1] = 0
		elif location[1] < 0:
			location[1] = hiveH-h



		locations += [location]


		h = hive[location[1]:h+location[1], location[0]:w+location[0]]
		h[mask!= 0] = beeDrawn[mask!=0]

	if lastLocations == None:
		return hive

	else:
		return hive, locations



if __name__ == "__main__":

	dir = "wood.jpg"
	hive = cv2.imread(dir)

	dir = "bee1.jpg"
	beeImg = cv2.imread(dir)

	dir = "spider.jpg"
	spiderImg = cv2.imread(dir)
	cv2.imshow("s", spiderImg)


	hive = generateRandomBees(hive.copy(), beeImg, numBees=5, spiderImg=spiderImg)
	cv2.imshow("h", hive)
	cv2.waitKey(0)

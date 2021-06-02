import cv2
import numpy as np 
from tracking2 import Tracker

csv = np.loadtxt("b.csv", delimiter = ",")

tracker = Tracker(2,200, 5)

video =  cv2.VideoCapture('TownCentreXVID.avi')
cv2.namedWindow('live', cv2.WINDOW_NORMAL)
cv2.namedWindow('SmokeNmirror', cv2.WINDOW_NORMAL)
pause = False
fr = 0
data = []
while(1):
	ret, frame = video.read()

	if ret: 

		data = csv[(csv[:,1]==fr)]
		copyFrame = frame.copy()
		detections= []

		for i in range(len(data)):
			det = [int(data[i][8]), int(data[i][9]), int(data[i][10]), int(data[i][11])]
			detections.append(np.array(det))

			cv2.rectangle(copyFrame,(det[0],det[1]),(det[2],det[3]),
				(0,255,0),2)

		if len(detections) > 0 :
			tracker.update(detections)

		# try: 
		for i in range(len(tracker.trackerList)):
			if len(tracker.trackerList[i].tracePath)>1 :
				for j in range(len(tracker.trackerList[i].tracePath)-1):
					x1 = tracker.trackerList[i].tracePath[j][0][0]
					y1 = tracker.trackerList[i].tracePath[j][1][0]
					x2 = tracker.trackerList[i].tracePath[j+1][0][0]
					y2 = tracker.trackerList[i].tracePath[j+1][1][0]
					# print(tracker.trackerList[i].tracePath[j][0])
					cv2.rectangle(frame, (int(x1),int(y1)), 
					(int(x1+5),int(y1+5)),tracker.trackerList[i].color, 5)

				k =tracker.trackerList[i].prediction
				cv2.rectangle(frame, (int(k[0]),int(k[1])), 
					(int(k[2]),int(k[3])),tracker.trackerList[i].color, 3)
		# except:
		# 	continue

		copyFrame = cv2.resize(copyFrame,(0,0),fx=0.25,fy=0.25)
		cv2.imshow("live",copyFrame)


		frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
		cv2.imshow("SmokeNmirror",frame)

		fr+=1

		k = cv2.waitKey(1) & 0xff
		if k == 27 :
			break
		if k == 112:  # 'p' has been pressed. this will pause/resume the code.
			pause = not pause
			if (pause is True):
				print("Code is paused. Press 'p' to resume..")
				while (pause is True):
			# stay in this loop until
					key = cv2.waitKey(30) & 0xff
					if key == 112:
						pause = False
						print("Resume code..!!")
						break
	else:
		break

video.release()
cv2.destroyAllWindows()	
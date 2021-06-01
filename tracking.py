import numpy as np
from misc import TrackUtils 
from scipy.optimize import linear_sum_assignment
from kalman import KalmanBoxTracker
from collections import deque
import uuid
import random




class Track(object):
	"""
	Track:

		trackId		: the id of the tracker
		KF 			: KalmanFIlter
		Prediction 	: Prediction of centroid for the next frame
		tracePath 	: Path histroy to make a trail of centroid
	
	"""
	def __init__(self, prediction):
		self.trackId = str(uuid.uuid4())
		self.KF = KalmanBoxTracker(prediction)
		self.prediction = np.asarray(prediction)
		self.undetectedFrameCount = 0
		self.tracePath = deque(maxlen=25)
		self.IOU_history = deque(maxlen=5)
		self.IOU_history.append(self.prediction)
		self.color= [random.randint(1,15)*15,random.randint(1,15)*15,
					random.randint(0,15)*15]




class Tracker(object):

	
	def __init__(self, thresh1, thresh2, thresh3 ):
		# super(Tracker, self).__init__()
		self.IOU_thresh = float(thresh1)
		self.dist_thresh= thresh2
		self.absent_frame_thresh = thresh3
		self.trackerList = []
		self.assignment = []
		self.utils = TrackUtils()

	

	def assign(self,cost):

		row_id, col_id = linear_sum_assignment(cost)
		self.assignment  = [-1 for i in range(len(self.trackerList))]

		for i in range(len(row_id)):
			self.assignment[row_id[i]] = col_id[i]



	def unassignment(self,detections):

		unassign_trackerList = []
	
		for i in range(len(self.assignment)):
			
			if (self.assignment[i] != -1):
				
				if (self.cost[i][self.assignment[i]]) > self.dist_thresh:
					self.assignment[i] = -1
					unassign_trackerList.append(i)
					self.trackerList[i].undetectedFrameCount += 1
				pass
				
			else:
				self.trackerList[i].undetectedFrameCount += 1

		return unassign_trackerList


	def __del__(self):
		del_trackList = []
		for i in range(len(self.trackerList)):
			if self.trackerList[i].undetectedFrameCount > self.absent_frame_thresh:
				del_trackList.append(i)

		if len(del_trackList) > 0:
			for idx in del_trackList:
				if idx < len(self.trackerList):
					del self.trackerList[idx]
					del self.assignment[idx]



	def update(self, detections):
		"""
		detections = [[0,0,0,0],[0,0,0,0]...]
	
		"""


		if len(self.trackerList)==0:
			for i in range(len(detections)):
				point = Track(detections[i])
				self.trackerList.append(point)


		self.cost = self.utils.calculate_cost(self.trackerList,detections)
		# get assignment according to hungarien algorithm
		self.assign(self.cost)
		# to check if different thresholds (IOU and Dist) are being satisfied
		unassign_trackerList = self.unassignment(detections)
		
		self.__del__()


		# for unassign detections
		for i in range(len(detections)):
			if i not in self.assignment:
				track = Track(detections[i])
				self.trackerList.append(track)


		# Update the kalman filter
		for i in range(len(self.assignment)):
			
			if self.assignment[i] != -1:
				self.trackerList[i].undetectedFrameCount = 0  
				self.trackerList[i].KF.update((detections[self.assignment[i]]+self.trackerList[i].KF.predict())/2)
				self.trackerList[i].prediction = self.trackerList[i].KF.predict()

			centroid_dot = self.utils.centroid(self.trackerList[i].prediction)
			self.trackerList[i].tracePath.append(centroid_dot)

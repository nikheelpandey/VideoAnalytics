import numpy as np 
from matplotlib import path


class TrackUtils(object):
	def __init__(self) -> None:
		super().__init__()
		#field of view 


	def centroid(self,arr):
		"""
		returns centroid of an array[x,y,x+w,,y+h]
		"""
		x = [arr[0]+arr[2]]
		y = [arr[1]+arr[3]]
		
		return (np.array([x,y])/2)


	def calculate_cost(self, trackerList, detections ):
		"""
		input: tracker list, detection_list
		returns the cost of assignment
		"""
		cost = np.zeros(shape=(len(trackerList), len(detections)))

		trackCent= [self.centroid(trackerList[i].prediction) for i in range(len(trackerList))]
		detectCent= [self.centroid(detections[i]) for i in range(len(detections))]
		
		for i in range(len(trackCent)):
			for j in range(len(detectCent)):

				diff =  trackCent[i]-detectCent[j] 
				distance = np.sqrt(diff[0][0]*diff[0][0] +
				diff[1][0]*diff[1][0])
				cost[i][j] = distance

		return (cost) 




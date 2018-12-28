from tkinter import *
from tkinter.ttk import *

import os
import sys
import cv2
import dlib
import imutils
import numpy as np

from time import time
from scipy.spatial import distance
from multiprocessing.dummy import Pool
from PIL import ImageTk, Image, ImageDraw, ImageFont
from imutils.video import VideoStream, FPS

def main():
	detector = dlib.get_frontal_face_detector()
	# sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
	# facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

	num_map = np.vectorize(distance.euclidean, signature='(n),(m)->()')

	cap = cv2.VideoCapture(0)   #'data/' + dirs[0]
	fps = FPS().start()
	counter = 0
	while True:
		frame = cap.read()[1]
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		fps.update()
		if counter % 50 == 0:
			trackers = []
			dets = detector(rgb, 1)
			for d in dets:
				tracker = dlib.correlation_tracker()
				tracker.start_track(rgb, d)
				trackers.append(tracker)

		elif len(trackers) > 0:
			for tracker in trackers:
				confidence = tracker.update(rgb)
				if confidence > 10:
					drect = tracker.get_position()
					left, top, right, bottom = tuple(map(int, (drect.left(), drect.top(), drect.right(), drect.bottom())))
					cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
		counter+=1
		cv2.imshow('img', frame)
		k = cv2.waitKey(1) & 0xff
		if k == ord('q'):
			break
	cap.release()
	fps.stop()
	print(f"{fps.fps():.2f}")

if __name__ == '__main__':
	main()
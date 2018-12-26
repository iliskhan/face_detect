import os
import dlib
import cv2 as cv
import numpy as np

from time import time 
from queue import Queue
from scipy.spatial import distance
from multiprocessing import Pool, Manager, Process, Queue

def main():

	detector = dlib.get_frontal_face_detector()
	sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
	facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
	
	manager = Manager()
	images_q = manager.Queue(25)

	Process(target=worker, args=(images_q,), daemon=True).start()

	Process(target)

	flag = True
	close = False
	
	while not close:
		while True:
			if not images_q.empty():
				img = images_q.get()
				break

		dets = detector(img, 1)
		if len(dets):
			track_windows = tuple((d.top(), d.height(), d.left(), d.width()) for d in dets)
			histograms = tuple(map(histogramic, track_windows, [img for i in track_windows]))
			for step in range(24):
				frame = images_q.get()
				hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

				i = 0
				for roi_hist, term_crit in histograms:
					dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)

					ret, track_window = cv.CamShift(dst, track_windows[i], term_crit)
					
					pts = cv.boxPoints(ret)
					pts = np.int0(pts)
					frame = cv.polylines(frame,[pts],True, 255,2)
					i+=1

				cv.imshow('img2',frame)
				k = cv.waitKey(60) & 0xff
				if k == 27:
					close = True
					break

		else:
			frame = images_q.get()
			cv.imshow('img2',frame)
			k = cv.waitKey(60) & 0xff
			if k == 27:
				close = True
				break
			

def histogramic(track_window, frame):

	r,h,c,w = track_window
	roi = frame[r:r+h, c:c+w]
	hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
	mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
	roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
	cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

	term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
	return roi_hist, term_crit

def worker(images_q):
	
	cap = cv.VideoCapture(0)
	while cap.isOpened():
		if images_q.full():
			images_q.get()
			images_q.put(cap.read()[1])
		else:
			images_q.put(cap.read()[1])

	cap.release()
	
if __name__ == '__main__':
	main()
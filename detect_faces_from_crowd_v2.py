import os
import dlib
import cv2 as cv
import numpy as np

from time import time 
from queue import Queue
from scipy.spatial import distance
from multiprocessing import Pool, Manager, Process, Queue

def main():
	FRAME_COUNT = 25
	
	manager = Manager()
	images_q = manager.Queue(FRAME_COUNT)
	points_array = manager.Queue(1)

	Process(target=worker, args=(images_q,), daemon=True).start()

	Process(target=det_and_rec, args=(images_q, points_array), daemon=True).start()

	close = False
	
	while not close:
		while True:
			if not images_q.empty():
				img = images_q.get()
				break

		if points_array.full():
			dets = points_array.get()

			if len(dets):
				track_windows = tuple((d.top(), d.height(), d.left(), d.width()) for d in dets)
				histograms = tuple(map(histogramic, track_windows, [img for i in track_windows]))
				for step in range(FRAME_COUNT-1):
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
		else:
			frame = images_q.get()
			cv.imshow('img2',frame)
			k = cv.waitKey(60) & 0xff
			if k == 27:
				close = True
				break
			
def det_and_rec(images_q, points_array):
	num_map = np.vectorize(distance.euclidean, signature='(n),(m)->()')

	detector = dlib.get_frontal_face_detector()
	sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
	facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

	matrix_descriptors = None
	while True:
		while True:
			if not images_q.empty():
				frame = images_q.get()
				break

		dets = detector(frame, 1)
		if points_array.full():
			points_array.get()
			points_array.put(dets)
		else:
			points_array.put(dets)

		for d in dets:
			shape = sp(frame, d)
			face_descriptor = facerec.compute_face_descriptor(frame, shape)
			face_descriptor = np.array([face_descriptor])
				
			if matrix_descriptors is not None:
				print(f'matrix_descriptors: {matrix_descriptors.shape}')

				vector_of_differences = num_map(face_descriptor,matrix_descriptors)
				index = np.argmin(vector_of_differences)

				if vector_of_differences[index] <= 0.7:
					cv.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (255,0,255), 2)

				else:
					
					cv.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0,0,255), 2)
					matrix_descriptors = np.append(matrix_descriptors, face_descriptor, axis=0)

			else:
				matrix_descriptors = face_descriptor

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
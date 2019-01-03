import os
import sys
import cv2
import dlib
import imutils
import numpy as np

from time import time, sleep
from scipy.spatial import distance
from multiprocessing import Process, Manager
from PIL import ImageTk, Image, ImageDraw, ImageFont
from imutils.video import VideoStream, FPS

def main():
	print('blas -'  , dlib.DLIB_USE_BLAS)
	print('cuda -'  , dlib.DLIB_USE_CUDA)
	print('lapack -', dlib.DLIB_USE_LAPACK)
	print('avx -'   , dlib.USE_AVX_INSTRUCTIONS)
	print('neon -'  , dlib.USE_NEON_INSTRUCTIONS)
	
	manager = Manager()

	count = manager.Value('i', 0)

	dets_q = manager.Queue(1)
	images_q = manager.Queue(25)
	q_for_detproc = manager.Queue(1)
	q_for_countproc = manager.Queue(1)

	Process(target=counting_process, args=(q_for_countproc, count), daemon=True).start()
	Process(target=capturing_process, args=(images_q, q_for_detproc), daemon=True).start()
	Process(target=detecting_process, args=(q_for_detproc, dets_q, q_for_countproc), daemon=True).start()

	counter = 0
	trackers = []

	while True:
		if images_q.qsize() > 10:
			frame = images_q.get()
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			if counter % 25 == 0:
				counter = 0
				if not dets_q.empty():
					trackers = []
					dets, rgb = dets_q.get()
					for d in dets:
						tracker = dlib.correlation_tracker()
						tracker.start_track(rgb, d)
						trackers.append(tracker)

			elif len(trackers) > 0:
				for tracker in trackers:
					confidence = tracker.update(rgb)
					if confidence > 8:
						drect = tracker.get_position()
						left, top, right, bottom = tuple(map(int, (drect.left(), drect.top(), drect.right(), drect.bottom())))
						cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
			counter+=1
			cv2.imshow('img', frame)
			k = cv2.waitKey(3) & 0xff
			if k == ord('q'):
				break

def counting_process(q_for_countproc, count):
	sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
	facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

	num_map = np.vectorize(distance.euclidean, signature='(n),(m)->()')

	matrix_of_descriptors = 0

	while True:
		if not q_for_countproc.empty():
			dets, rgb = q_for_countproc.get()
			for d in dets:
				shape = sp(rgb, d)
				face_descriptor = facerec.compute_face_descriptor(rgb, shape)
				face_descriptor = np.array([face_descriptor])

				if type(matrix_of_descriptors) is int:
					matrix_of_descriptors = face_descriptor
				else:
					vector_of_differences = num_map(face_descriptor,matrix_of_descriptors)
					index = np.argmin(vector_of_differences)
					if vector_of_differences[index] >= 0.6:
						matrix_of_descriptors = np.append(matrix_of_descriptors, face_descriptor, axis=0)
			if type(matrix_of_descriptors) is not int:
				print(len(matrix_of_descriptors))

def capturing_process(images_q, q_for_detproc):
	cap = cv2.VideoCapture('rep.mp4')
	while cap.isOpened():
		img = imutils.resize(cap.read()[1], width=1000)
		# if images_q.full():
		# 	images_q.get()
		# 	images_q.put(img)
		# else:
		images_q.put(img)
		if q_for_detproc.empty():
			q_for_detproc.put(img)

	cap.release()

def detecting_process(q_for_detproc, dets_q, q_for_countproc):

	detector = dlib.get_frontal_face_detector()
	
	while True:
		while not q_for_detproc.empty():
			rgb = cv2.cvtColor(q_for_detproc.get(), cv2.COLOR_BGR2RGB)
			lt = time()
			dets = detector.run(rgb,1)
			
			for d, conf, orient in zip(*dets):

				if conf < 1:

					dets[0].remove(d)
					dets[1].remove(conf)
					dets[2].remove(orient)

			print(time()-lt)
			if dets_q.full():
				dets_q.get()
				dets_q.put((dets[0], rgb))
			else:
				dets_q.put((dets[0], rgb))

			if q_for_countproc.full():
				q_for_countproc.get()
				q_for_countproc.put((dets[0], rgb))
			else:
				q_for_countproc.put((dets[0], rgb))

if __name__ == '__main__':
	main()
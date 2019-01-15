import os
import sys
import cv2
import dlib
import imutils
import numpy as np
import threading

from time import time, sleep
from scipy.spatial import distance
from imutils.video import VideoStream, FPS
from multiprocessing import Process, Manager, Value
from PIL import ImageTk, Image, ImageDraw, ImageFont


def main():
	print('blas -'  , dlib.DLIB_USE_BLAS)
	print('cuda -'  , dlib.DLIB_USE_CUDA)
	print('lapack -', dlib.DLIB_USE_LAPACK)
	print('avx -'   , dlib.USE_AVX_INSTRUCTIONS)
	print('neon -'  , dlib.USE_NEON_INSTRUCTIONS)
	
	manager = Manager()

	#количество лиц
	count = Value('i', 0)

	time_det = Value('d', 0.0)
	time_count = Value('d', 0.0)

	dets_q = manager.Queue(1)
	images_q = manager.Queue(25)

	#очередь для процесса детектирования
	q_for_detproc = manager.Queue(1)

	#очередь для процесса распознования и подсчета
	q_for_countproc = manager.Queue(1)

	Process(target=counting_process, args=(q_for_countproc, count , time_count), daemon=True).start()
	Process(target=capturing_process, args=(images_q, q_for_detproc), daemon=True).start()
	Process(target=detecting_process, args=(q_for_detproc, dets_q, q_for_countproc, time_det), daemon=True).start()
	font = cv2.FONT_HERSHEY_SIMPLEX
	counter = 0
	trackers = []

	while True:
		if not images_q.empty():
			frame = images_q.get()
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			if counter % 50 == 0:
				counter = 0
				trackers = []
				if not dets_q.empty():
					
					dets, rgb = dets_q.get()
					for d in dets:
						tracker = dlib.correlation_tracker()
						tracker.start_track(rgb, d)
						trackers.append(tracker)

			elif len(trackers) > 0:
				for tracker in trackers:
					confidence = tracker.update(rgb)
					if confidence > 7:
						drect = tracker.get_position()
						left, top, right, bottom = tuple(map(int, (drect.left(), drect.top(), drect.right(), drect.bottom())))
						cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)

			counter+=1
			height, width = frame.shape[:2]
			cv2.putText(frame, str(count.value), (width-100, height-100) , font, 4,(255,255,255),2,cv2.LINE_AA)
			cv2.imshow('img', frame)
			k = cv2.waitKey(3) & 0xff
			if k == ord('q'):
				break

			print('detecting time = ', time_det.value)
			print('counting time = ', time_count.value)

def counting_process(q_for_countproc, count, time_count):
	sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
	facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

	num_map = np.vectorize(distance.euclidean, signature='(n),(m)->()')

	matrix_of_descriptors = 0

	while True:
		
		if not q_for_countproc.empty():
			dets, rgb = q_for_countproc.get()
			for d in dets:
				lt = time()
				shape = sp(rgb, d)
				face_descriptor = facerec.compute_face_descriptor(rgb, shape)
				time_count.value = time() - lt
				face_descriptor = np.array([face_descriptor])
				rgb_copy = rgb[d.top():d.bottom(), d.left():d.right()]
				rgb_copy = cv2.cvtColor(rgb_copy, cv2.COLOR_RGB2BGR)

				if type(matrix_of_descriptors) is int:
					matrix_of_descriptors = face_descriptor
					cv2.imwrite(f'{len(matrix_of_descriptors)}.jpg', rgb_copy)
					count.value = len(matrix_of_descriptors)
				else:
					vector_of_differences = num_map(face_descriptor,matrix_of_descriptors)

					index = np.argmin(vector_of_differences)
					if vector_of_differences[index] >= 0.6:

						matrix_of_descriptors = np.append(matrix_of_descriptors, face_descriptor, axis=0)
						
						cv2.imwrite(f'{len(matrix_of_descriptors)}.jpg', rgb_copy)
						count.value = len(matrix_of_descriptors)

def capturing_process(images_q, q_for_detproc):
	cap = cv2.VideoCapture(0)
	while cap.isOpened():
		#print('capturing')
		img = cap.read()[1]
		img = imutils.resize(img, width=1000)
		if images_q.full():
			images_q.get()
			images_q.put(img)
		else:
			images_q.put(img)
		if q_for_detproc.empty():
			q_for_detproc.put(img)

	cap.release()

def detecting_process(q_for_detproc, dets_q, q_for_countproc, time_det):

	detector = dlib.get_frontal_face_detector()
	
	while True:

		lt = time()
		if not q_for_detproc.empty():
			rgb = cv2.cvtColor(q_for_detproc.get(), cv2.COLOR_BGR2RGB)
			#lt = time()
			dets = detector.run(rgb,1,1)
			
			for d, conf, orient in zip(*dets):

				if orient != 0:

					dets[0].remove(d)
					dets[1].remove(conf)
					dets[2].remove(orient)

			#print(time()-lt)
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
			time_det.value = time() - lt

if __name__ == '__main__':
	main()
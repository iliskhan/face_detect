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
	print('blas',dlib.DLIB_USE_BLAS)
	print('cuda',dlib.DLIB_USE_CUDA)
	print('lapack',dlib.DLIB_USE_LAPACK)
	print('avx',dlib.USE_AVX_INSTRUCTIONS)
	print('neon',dlib.USE_NEON_INSTRUCTIONS)
	
	manager = Manager()
	images_q = manager.Queue(25)
	dets_q = manager.Queue(1)

	Process(target=capturing_process, args=(images_q,), daemon=True).start()
	Process(target=detecting_process, args=(images_q, dets_q), daemon=True).start()

	num_map = np.vectorize(distance.euclidean, signature='(n),(m)->()')

	fps = FPS().start()
	trackers = []
	counter = 0
	while True:
		if not images_q.empty() and not images_q.qsize() < 10:
			frame = images_q.get()
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			fps.update()

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
	fps.stop()
	print(f"fps = {fps.fps():.2f}")

def capturing_process(images_q):
	cap = cv2.VideoCapture('rep.mp4')
	while cap.isOpened():
		img = imutils.resize(cap.read()[1], width=700)
		# if images_q.full():
		# 	images_q.get()
		# 	images_q.put(img)
		# else:
		images_q.put(img)

	cap.release()

def detecting_process(images_q, dets_q):
	detector = dlib.get_frontal_face_detector()
	sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
	facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
	while True:
		while not images_q.empty():
			rgb = cv2.cvtColor(images_q.get(), cv2.COLOR_BGR2RGB)
			lt = time()
			dets = detector.run(rgb,2,1)
			
			for d, conf, orient in zip(*dets):

				if conf < 1:
					print(conf)
					dets[0].remove(d)
					dets[1].remove(conf)
					dets[2].remove(orient)
			print(time()-lt)
			if dets_q.full():
				dets_q.get()
				dets_q.put((dets[0], rgb))
			else:
				dets_q.put((dets[0], rgb))	

if __name__ == '__main__':
	main()
import dlib
import cv2
import numpy as np
import os

from queue import Queue
from scipy.spatial import distance
from multiprocessing import Pool, Manager, Process, Queue

def worker(images_q):
	cap = cv2.VideoCapture(0)
	for i in range(10):
		img = cap.read()[1]
		images_q.put(img)
	cap.release()

def main():

	detector = dlib.get_frontal_face_detector()
	sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
	facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
	manager = Manager()
	images_q = manager.Queue(25)

	p = Process(target=worker, args=(images_q,), daemon=True)
	p.start()
	p.join()
	for i in range(10):
		images_q.get()
		print(i)

if __name__ == '__main__':
	main()
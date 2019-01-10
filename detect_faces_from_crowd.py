import dlib
import cv2
import numpy as np
import os
import imutils

from scipy.spatial import distance

sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

num_map = np.vectorize(distance.euclidean, signature='(n),(m)->()')
cap = cv2.VideoCapture('rep.mp4')

while(cap.isOpened()):
	ret, frame = cap.read()

	if ret:
		frame = imutils.resize(frame, width=600)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		dets = detector.run(rgb)
		if len(dets[0]) != 0:
			
			for d, conf, orient in zip(*dets):
					print('1 ---', orient)
			# if orient == 2:
					
			cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0,0,255), 2)

	elif not ret:
		break
	cv2.imshow('frame',frame)
	k = cv2.waitKey(3) & 0xff
	if k == 27:
		break


cap.release()
out.release()
cv2.destroyAllWindows()
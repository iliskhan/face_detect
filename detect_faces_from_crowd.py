import dlib
import cv2
import numpy as np
import os

from scipy.spatial import distance

sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

num_map = np.vectorize(distance.euclidean, signature='(n),(m)->()')

font = cv2.FONT_HERSHEY_SIMPLEX

dirs = os.listdir('data')

cap = cv2.VideoCapture(0) #f'data/{dirs[1]}'
fourcc = cv2.VideoWriter_fourcc(*'XVID')

if cap.isOpened():
	w = int(cap.get(3))
	h = int(cap.get(4))
out = cv2.VideoWriter('res/output.mp4', int(cap.get(6)), 15.0, (w,h))

matrix_descriptors = None
w = int(w * 0.70)
h = int(h * 0.92)

while(cap.isOpened()):
	ret, frame = cap.read()

	if ret:
		dets = detector(frame, 2)

		for d in dets:

			shape = sp(frame, d)
		
			face_descriptor = facerec.compute_face_descriptor(frame, shape)
			face_descriptor = np.array([face_descriptor])
			
			if matrix_descriptors is not None:
				print(f'matrix_descriptors: {matrix_descriptors.shape}')

				vector_of_differences = num_map(face_descriptor,matrix_descriptors)
				index = np.argmin(vector_of_differences)

				if vector_of_differences[index] <= 0.7:
					cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (255,0,255), 2)

				else:
					
					cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0,0,255), 2)
					matrix_descriptors = np.append(matrix_descriptors, face_descriptor, axis=0)

			else:
				matrix_descriptors = face_descriptor
		cv2.putText(frame, f'Unic: {0 if matrix_descriptors is None else matrix_descriptors.shape[0]}', (w, h), font, 1, (255,0,0), 2, cv2.LINE_AA)
		#out.write(frame)
		print('frame записан')


	elif not ret:
		break
	cv2.imshow('frame',frame)
	k = cv2.waitKey(300) & 0xff
	if k == 27:
		break

	for yg in range(4):
		cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
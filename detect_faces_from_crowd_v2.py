import os
import cv2
import dlib
import imutils
import numpy as np

from time import time
from multiprocessing import Pool
from scipy.spatial import distance

detector = dlib.get_frontal_face_detector()

sp = dlib.shape_predictor(
	'models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1(
	'models/dlib_face_recognition_resnet_model_v1.dat')

def img_croper(cap):

	frame = cap.read()[1]

	height, width = frame.shape[:2]
	h_half, w_half = height//2, width//2 
	h_coords, w_coords = [0, h_half, height], [0, w_half, width]
	while True:
		frame = cap.read()[1]
		image_chunks = tuple(frame[h_coords[i]:h_coords[i+1],
								     w_coords[j]:w_coords[j+1]] 
								     	for i in range(len(h_coords)-1) 
								     	for j in range(len(w_coords)-1))

		# left_top = frame[h_coords[0]:h_coords[1], w_coords[0]:w_coords[1]]
		# right_top = frame[h_coords[0]:h_coords[1], w_coords[1]:w_coords[2]]
		# left_bottom = frame[h_coords[1]:h_coords[2], w_coords[0]:w_coords[1]]
		# right_bottom = frame[h_coords[1]:h_coords[2], w_coords[1]:w_coords[2]]

		yield image_chunks, frame

def process_face_detect(frame):

	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	dets = detector.run(rgb,0,1)
	face_descriptors = []
	if len(dets[0]) != 0:
		
		for d, conf, orient in zip(*dets):
			if orient != 0:
				dets[0].remove(d)
				dets[1].remove(conf)
				dets[2].remove(orient)
				continue
			
			shape = sp(rgb, d)

			face_descriptor = facerec.compute_face_descriptor(rgb, shape)
			face_descriptors.append(face_descriptor)
	return face_descriptors, dets[0]

def main():

	print('blas -'  , dlib.DLIB_USE_BLAS)
	print('cuda -'  , dlib.DLIB_USE_CUDA)
	print('lapack -', dlib.DLIB_USE_LAPACK)
	print('avx -'   , dlib.USE_AVX_INSTRUCTIONS)
	print('neon -'  , dlib.USE_NEON_INSTRUCTIONS)

	font = cv2.FONT_HERSHEY_SIMPLEX

	num_map = np.vectorize(distance.euclidean, signature='(n),(m)->()')
	cap = cv2.VideoCapture(1)
	cap.set(3,1500)

	frame = cap.read()[1]
	height, width = frame.shape[:2]

	matrix_of_descriptors = 0
	count = 0
	with Pool() as p:
		
		for image_chunks, frame in img_croper(cap):
			
			data = p.map(process_face_detect, image_chunks)
			for face_descriptors, dets in data:
				for i in range(len(face_descriptors)):
					face_descriptor = face_descriptors[i]
					d = dets[i]

					face_descriptor = np.array([face_descriptor])
					rgb_copy = frame[d.top():d.bottom(), d.left():d.right()]

					if type(matrix_of_descriptors) is int:
						matrix_of_descriptors = face_descriptor
						cv2.imwrite(f'{len(matrix_of_descriptors)}.jpg', rgb_copy)
						count+=1

					else:
						vector_of_differences = num_map(face_descriptor,matrix_of_descriptors)

						index = np.argmin(vector_of_differences)
						if vector_of_differences[index] >= 0.6:

							matrix_of_descriptors = np.append(matrix_of_descriptors, face_descriptor, axis=0)
							count+=1
							
							cv2.imwrite(f'{len(matrix_of_descriptors)}.jpg', rgb_copy)

			cv2.putText(frame, str(count), (width-100, height-100) , font, 4,(255,255,255),2,cv2.LINE_AA)
			cv2.imshow('frame',frame)
			k = cv2.waitKey(3) & 0xff
			if k == 27:
				break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
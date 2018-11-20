import dlib
import cv2
import numpy as np
import os
import sys

from scipy.spatial import distance
from multiprocessing import Pool, Manager

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

def video_maker(frames_with_data):


	frame, w, h = frames_with_data

	dets = detector(frame, 1)
	
	descriptors = []

	for d in dets:
		shape = sp(frame, d)
		face_descriptor = facerec.compute_face_descriptor(frame, shape)
		
		descriptors.append(face_descriptor)
			
		cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (255,0,0), 2)

	return frame, descriptors

def main():

	num_map = np.vectorize(distance.euclidean, signature='(n),(m)->()')

	#matrix_descriptors = None

	dirs = os.listdir('data')

	cap = cv2.VideoCapture(0)    #'data/' + dirs[0]

	NUM_FRAMES = 200 #int(cap.get(7))
	#CODEC = int(cap.get(6))

	if cap.isOpened():
		w = int(cap.get(3))
		h = int(cap.get(4))

	#out = cv2.VideoWriter('res/output6.mp4', CODEC, 25.0, (w,h))

	w = int(w * 0.70)
	h = int(h * 0.92)

	frames_with_data = (d for i, d in enumerate(
							((cap.read()[1], w, h) for y in range(NUM_FRAMES))) 
							if i % 8 == 0)

	font = cv2.FONT_HERSHEY_SIMPLEX
	matrix_descriptors = np.array([])

	with Pool() as p:
		for d, i in enumerate(p.imap(video_maker, frames_with_data, chunksize=1)):

			frame = i[0]

			for y in i[1]:
				face_descriptor = np.array([y])
				
				if len(matrix_descriptors) > 0:
					
					vector_of_differences = num_map(face_descriptor,matrix_descriptors)
					index = np.argmin(vector_of_differences)

					if vector_of_differences[index] >= 0.6:
						matrix_descriptors = np.append(matrix_descriptors, face_descriptor, axis=0)

				elif len(matrix_descriptors) == 0:

					matrix_descriptors = face_descriptor
			cv2.putText(frame, 
						f'Unic: {len(matrix_descriptors)}', 
						(w, h), 
						font, 
						1, 
						(0,0,255), 
						2, 
						cv2.LINE_AA)					

			cv2.imshow('img',frame)
			k = cv2.waitKey(1) & 0xff

			if k == 27:
				break
				sys.exit()
			#out.write(i)

			print(f'кадр {d} записан')
		p.close()
		p.join()

	cap.release()
	#out.release()

if __name__ == '__main__':
	main()
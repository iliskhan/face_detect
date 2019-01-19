import os
import cv2
import dlib
import imutils
import numpy as np

from time import time
from scipy.spatial import distance

def main():
	sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
	facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
	detector = dlib.get_frontal_face_detector()

	print('blas -'  , dlib.DLIB_USE_BLAS)
	print('cuda -'  , dlib.DLIB_USE_CUDA)
	print('lapack -', dlib.DLIB_USE_LAPACK)
	print('avx -'   , dlib.USE_AVX_INSTRUCTIONS)
	print('neon -'  , dlib.USE_NEON_INSTRUCTIONS)

	font = cv2.FONT_HERSHEY_SIMPLEX

	num_map = np.vectorize(distance.euclidean, signature='(n),(m)->()')
	cap = cv2.VideoCapture(0)

	matrix_of_descriptors = 0
	count = 0
	while(cap.isOpened()):
		ret, frame = cap.read()
		lt = time()
		if ret:
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			dets = detector.run(rgb,0,1)
			if len(dets[0]) != 0:
				
				for d, conf, orient in zip(*dets):
					if orient != 0:
						dets[0].remove(d)
						dets[1].remove(conf)
						dets[2].remove(orient)
						continue
						
					cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0,0,255), 2)

					shape = sp(rgb, d)
					face_descriptor = facerec.compute_face_descriptor(rgb, shape)

					face_descriptor = np.array([face_descriptor])
					rgb_copy = rgb[d.top():d.bottom(), d.left():d.right()]
					rgb_copy = cv2.cvtColor(rgb_copy, cv2.COLOR_RGB2BGR)

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


		elif not ret:
			break

		print(time() - lt)
		height, width = frame.shape[:2]
		cv2.putText(frame, str(count), (width-100, height-100) , font, 4,(255,255,255),2,cv2.LINE_AA)
		cv2.imshow('frame',frame)
		k = cv2.waitKey(3) & 0xff
		if k == 27:
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
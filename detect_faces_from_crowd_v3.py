import dlib
import cv2
import numpy as np
import os

from scipy.spatial import distance
from multiprocessing import Pool , Manager 

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

num_map = np.vectorize(distance.euclidean, signature='(n),(m)->()')

font = cv2.FONT_HERSHEY_SIMPLEX

def video_maker(frames_with_data):
	
	frame, w, h, q1, q2 = frames_with_data

	dets = detector(frame, 1)
	matrix_descriptors = q2.get(True)
	
	for d in dets:
		shape = sp(frame, d)
		face_descriptor = facerec.compute_face_descriptor(frame, shape)

		face_descriptor = np.array([face_descriptor])

		if matrix_descriptors is not None:
			vector_of_differences = num_map(face_descriptor,matrix_descriptors)
			index = np.argmin(vector_of_differences)

			if vector_of_differences[index] <= 0.6:
				cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (255,0,0), 2)
			else:
				unic = q1.get(True)
				unic+=1
				q1.put(unic, True)
				cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0,0,255), 2)
				matrix_descriptors = np.append(matrix_descriptors, face_descriptor, axis=0)

		else:

			matrix_descriptors = face_descriptor
	q2.put(matrix_descriptors, True)
	unic = q1.get(True)
	cv2.putText(frame, f'Unic: {unic}', (w, h), font, 2, (0,0,255), 2, cv2.LINE_AA)
	q1.put(unic, True)
	return frame

def main():
	unic = 0
	matrix_descriptors = None
	manager = Manager()
	q1 = manager.Queue()
	q1.put(unic)

	q2 = manager.Queue()
	q2.put(matrix_descriptors)

	dirs = os.listdir('data')

	cap = cv2.VideoCapture('data/' + dirs[0])

	NUM_FRAMES = int(cap.get(7))
	CODEC = int(cap.get(6))

	if cap.isOpened():
		w = int(cap.get(3))
		h = int(cap.get(4))

	out = cv2.VideoWriter('res/output6.mp4', CODEC, 25.0, (w,h))

	w = int(w * 0.70)
	h = int(h * 0.92)

	frames_with_data = (d for i, d in enumerate(
							((cap.read()[1], w, h, q1, q2) for y in range(NUM_FRAMES))) 
							if i % 25 == 0)

	with Pool() as p:
		for d, i in enumerate(p.imap(video_maker, frames_with_data, chunksize=10)):
			
			out.write(i)

			print(f'кадр {d} записан')
		p.close()
		p.join()

	cap.release()
	out.release()

if __name__ == '__main__':
	main()
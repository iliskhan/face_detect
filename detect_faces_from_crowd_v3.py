from tkinter import *
from tkinter.ttk import *

import os
import sys
import cv2
import dlib
import imutils
import numpy as np
import threading

from queue import Queue
from scipy.spatial import distance
from multiprocessing import Pool, Manager
from PIL import ImageTk, Image, ImageDraw, ImageFont

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')


def img_generator(cap, w, h):
	while cap.isOpened():

		yield cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB), w, h

def video_maker(frames_with_data):


	frame, w, h = frames_with_data

	dets = detector(frame, 1)
	
	descriptors = []

	for d in dets:
		shape = sp(frame, d)
		face_descriptor = facerec.compute_face_descriptor(frame, shape)
		
		descriptors.append(face_descriptor)
			
		cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0,0,255), 2)

	return frame, descriptors

def propageter(q, cap):
	c.get(cap.read()[1])

def main():

	q = Queue(25)
	num_map = np.vectorize(distance.euclidean, signature='(n),(m)->()')

	cap = cv2.VideoCapture(0)    #'data/' + dirs[0]
	# cap.set(3,1080)
	# cap.set(4,720)
	#out = cv2.VideoWriter('res/output6.mp4', CODEC, 25.0, (w,h))
	img = imutils.resize(cap.read()[1], height=720)
	h, w = img.shape[:2]
	w = int(w * 0.7)
	h = int(h * 0.9)

	fnt = ImageFont.truetype("arial.ttf", 35)
	font = cv2.FONT_HERSHEY_SIMPLEX
	matrix_descriptors = np.array([])
	window = Tk()
	window.title('Счетчик')

	with Pool() as p:
		for i in p.imap(video_maker, img_generator(cap,w,h), chunksize=1):
			for child in window.winfo_children():
				child.destroy()

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

			frame = Image.fromarray(imutils.resize(frame, height=720))
			draw = ImageDraw.Draw(frame)

			draw.text((w, h), f'Уникальных: {len(matrix_descriptors)}', font=fnt, fill=(255,0,0))

			photo = ImageTk.PhotoImage(frame)

			label_with_image = Label(window, image=photo)
			label_with_image.image = photo

			ok_button = Button(window,
						text="Остановить",
						command=lambda: cap.release())

			label_with_image.pack()
			ok_button.pack(padx=40,pady=10,side=LEFT)
			window.update()
			
		p.close()
		p.join()

	cap.release()
	#out.release()
	window.mainloop()

if __name__ == '__main__':
	main()
from tkinter import *
from tkinter.ttk import *

import cv2
import PIL.Image, PIL.ImageTk
import time

import os
import sys
import cv2
import dlib
import imutils
import numpy as np

from scipy.spatial import distance
from multiprocessing import Pool, Manager
from PIL import ImageDraw, ImageFont

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
			
		cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0,0,255), 2)

	return frame, descriptors

class App:
	def __init__(self, window, window_title, video_source=0):
		self.window = window
		self.window.title(window_title)
		self.video_source = video_source

		self.vid = MyVideoCapture(self.video_source)

		self.canvas = Canvas(window, width = self.vid.width, height = self.vid.height)
		self.canvas.pack()

		self.btn_stop=Button(window, text="Остановить", width=50, command=self.vid.stop)
		self.btn_stop.pack(anchor=CENTER, expand=True)


		#self.delay = 15
		self.update()

		self.window.mainloop()

	def update(self):

		with Pool() as p:
			for d, i in enumerate(p.imap(video_maker, self.vid.img_generator(self.vid, self.vid.w, self.vid.h), chunksize=1)):
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

				frame = Image.fromarray(frame)
				draw = ImageDraw.Draw(frame)

				draw.text((w, h), f'Уникальных: {len(matrix_descriptors)}', font=fnt, fill=(255,0,0))

				photo = ImageTk.PhotoImage(frame)

				self.canvas.create_image(0, 0, image = photo, anchor = NW)

				self.window.update()


class MyVideoCapture:
	def __init__(self, video_source=0):

		self.vid = cv2.VideoCapture(video_source)
		if not self.vid.isOpened():
			raise ValueError("Unable to open video source", video_source)

		self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
		self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
		img = imutils.resize(self.vid.read()[1], height=720)
		h, w = img.shape[:2]
		self.w = int(w * 0.7)
		self.h = int(h * 0.9)

	def img_generator(self, vid, w, h):
		while vid.isOpened():

			yield cv2.cvtColor(vid.read()[1], cv2.COLOR_BGR2RGB), w, h

	def stop(self):
		self.vid.release()

	def __del__(self):
		if self.vid.isOpened():
			self.vid.release()
 
App(Tk(), "Tkinter")
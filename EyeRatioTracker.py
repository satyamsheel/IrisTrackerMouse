import cv2
import os
import numpy as np
import dlib
import math
from skimage.morphology import disk
from skimage.filters import rank
import json
import pyautogui
from datetime import datetime

#histogram equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

#face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def threshold_max(o_image, percent) :
	image = np.copy(o_image)
	flat = image.flatten()
	flat = np.sort(flat)
	index = np.searchsorted(flat, 40)
	flat = flat[:index]
	nums = int(percent * flat.shape[0])
	pixel_value = flat[-nums]

	for y in range(o_image.shape[0]) :
		for x in range(o_image.shape[1]):
			if o_image[y][x] > pixel_value :
				o_image[y][x] = 255
			else :
				o_image[y][x] = 0
	return o_image

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def equalize(image) :
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cl1 = clahe.apply(image)
	return cl1

def get_centroid(original_image, x_min, y_min, x_max, y_max) :
	
	x = 0
	y = 0

	image = original_image[y_min:y_max, x_min:x_max]
	count = 0
	for j in range(image.shape[0]) :
		for i in range(image.shape[1]) :
			if image[j, i] != 255 :
				count += 1
				x += i
				y += j
	if count == 0 :
		return None
	x //= count
	y //= count

	x += x_min
	y += y_min 

	image = cv2.circle(original_image, (x, y), 0, (255, 0, 0), 1)

	return image, (x, y)

def get_pupils(image, shape) :
	left_pupil = -1
	right_pupil = -1

	left_x_min = shape[36][0]
	left_x_max = shape[39][0]
	left_y_min, left_y_max = min(shape[37][1], shape[38][1]), max(shape[41][1], shape[40][1])
	right_x_min = shape[42][0]
	right_x_max = shape[45][0]
	right_y_min, right_y_max = min(shape[43][1], shape[44][1]), max(shape[46][1], shape[47][1])
	# left_eye = cv2.medianBlur(left_eye, 5)
	#left_eye = local_histogram_equalization(left_eye)
	#left_eye = apply_thresholding(left_eye)

	mask = np.zeros((image.shape[0], image.shape[1]))
	cv2.fillConvexPoly(mask, shape[36:41], 1)
	mask = mask.astype(np.bool)
	mask2 = np.zeros((image.shape[0], image.shape[1]))
	cv2.fillConvexPoly(mask2, shape[42:48], 1)
	mask2 = mask2.astype(np.bool)

	out = np.zeros_like(image)
	out.fill(255)
	out[mask] = image[mask]
	out[mask2] = image[mask2]

	out[left_y_min:left_y_max, left_x_min:left_x_max] = threshold_max(out[left_y_min:left_y_max, left_x_min:left_x_max], 0.10)
	out[right_y_min:right_y_max, right_x_min:right_x_max] = threshold_max(out[right_y_min:right_y_max, right_x_min:right_x_max], 0.10)
	left_image, left_pupil = get_centroid(out, left_x_min, left_y_min, left_x_max, left_y_max)
	right_image, right_pupil = get_centroid(out, right_x_min, right_y_min, right_x_max, right_y_max)

	return left_pupil, right_pupil, left_image, right_image

def distance(a, b) :
	x = (a[0] - b[0]) * (a[0] - b[0])
	y = (a[1] - b[1]) * (a[1] - b[1])
	ans = math.sqrt(x + y)
	return ans

def get_EAR(shape) :
	left_ear = (distance(shape[37], shape[41]) + distance(shape[38], shape[40])) / distance(shape[36], shape[39])
	right_ear = (distance(shape[43], shape[47]) + distance(shape[44], shape[46])) / distance(shape[42], shape[45])
	return left_ear, right_ear

def get_pupil_ratio(pupil_location, landmark_locations) :

	pupil_ratio = []
	sum_dist = 0
	for (x, y) in landmark_locations :
		dist = abs(pupil_location[0] - x) + abs(pupil_location[1] - y)
		pupil_ratio.append(1 / dist)
		sum_dist += dist
	pupil_ratio = np.array(pupil_ratio)
	pupil_ratio *= sum_dist
	return pupil_ratio


def detect_face(image) :

	dets = detector(image, 1)
	left_ear = None
	right_ear = None
	left_pupil_ratio = None
	right_pupil_ratio = None
	ret_dict = None
	for k, d in enumerate(dets) :
		shape = predictor(image, d)
		shape = shape_to_np(shape)
		left_ear, right_ear = get_EAR(shape)
		left_pupil, right_pupil, left_image, right_image = get_pupils(image, shape)
		left_pupil_ratio = get_pupil_ratio(left_pupil, shape[36:42])
		right_pupil_ratio = get_pupil_ratio(right_pupil, shape[42:48])

		ret_dict = {
			"left_ear" : left_ear,
			"right_ear" : right_ear,
			"left_pupil_ratio" :left_pupil_ratio,
			"right_pupil_ratio" : right_pupil_ratio,
			"left_image" : left_image,
			"right_image" : right_image
		}

	return image, ret_dict

def pipeline(image) :
	try :	
		image = equalize(image)
		image, ret_dict = detect_face(image)
		return image, ret_dict
	except Exception as e:
		print(e)
		return None, None

def main() :

	cam = cv2.VideoCapture(0)
	cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	for i in range(50) :
		cam.read()

	while True :
		ret, image = cam.read()
		face, ret_dict = pipeline(image)

		if ret_dict is None :
			continue

		left_ear = ret_dict["left_ear"]
		right_ear = ret_dict["right_ear"]

		cv2.imwrite("project data/face_images/image_" + str(datetime.now().timestamp()) + ".jpg", ret_dict["left_image"])
		cv2.imwrite("project data/face_images/image_" + str(datetime.now().timestamp()) + ".jpg", ret_dict["right_image"])
		print(pyautogui.position(), left_ear, right_ear)
			

if __name__ == "__main__" :
	main()
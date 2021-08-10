import cv2
import os
import numpy as np
import dlib
import math
from skimage.morphology import disk
from skimage.filters import rank
import json

#histogram equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

#face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def apply_thresholding(img, th_type = "OTSU") :
	th2 = None
	if th_type == "OTSU" :
		blur = cv2.GaussianBlur(img,(5,5),0)
		_,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	else :
		th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)	
	return th2

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
	# cl1 = clahe.apply(image)
	return image

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

	return (x, y)

def contrast_stretch(image) :
	min_val = np.min(image)
	max_val = np.max(image)
	for y in range(image.shape[0]) :
		for x in range(image.shape[1]) :
			pixel = image[y][x]
			image[y][x] = ((pixel - min_val) / (max_val - min_val))*255
	return image

def get_pupils(image, shape) :
	left_pupil = -1
	right_pupil = -1

	left_x_min = shape[36][0]
	left_x_max = shape[39][0]
	left_y_min, left_y_max = min(shape[37][1], shape[38][1]), max(shape[41][1], shape[40][1])
	right_x_min = shape[42][0]
	right_x_max = shape[45][0]
	right_y_min, right_y_max = min(shape[43][1], shape[44][1]), max(shape[46][1], shape[47][1])

	image[left_y_min:left_y_max, left_x_min:left_x_max] = local_histogram_equalization(image[left_y_min:left_y_max, left_x_min:left_x_max])
	image[right_y_min:right_y_max, right_x_min:right_x_max] = local_histogram_equalization(image[right_y_min:right_y_max, right_x_min:right_x_max])
	image[left_y_min:left_y_max, left_x_min:left_x_max] = contrast_stretch(image[left_y_min:left_y_max, left_x_min:left_x_max])
	image[right_y_min:right_y_max, right_x_min:right_x_max] = contrast_stretch(image[right_y_min:right_y_max, right_x_min:right_x_max])
	cv2.imshow("image", image)
	cv2.waitKey(0)
	cv2.imshow("image", image)
	cv2.waitKey(0)
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
	_, out = cv2.threshold(out, 15, 255, cv2.THRESH_BINARY)
	out = cv2.medianBlur(out, 5)

	left_pupil = get_centroid(out, left_x_min, left_y_min, left_x_max, left_y_max)
	right_pupil = get_centroid(out, right_x_min, right_y_min, right_x_max, right_y_max)

	return left_pupil, right_pupil


def local_histogram_equalization(image) :
	selem = disk(20)
	cl1 = rank.equalize(image, selem = selem)
	return cl1

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
	for k, d in enumerate(dets) :
		shape = predictor(image, d)
		shape = shape_to_np(shape)
		left_ear, right_ear = get_EAR(shape)
		left_pupil, right_pupil = get_pupils(image, shape)
		left_pupil_ratio = get_pupil_ratio(left_pupil, shape[36:42])
		right_pupil_ratio = get_pupil_ratio(right_pupil, shape[42:48])

	return image, left_ear, right_ear, left_pupil_ratio, right_pupil_ratio, shape[36: 42], shape[42:48]

def pipeline(image) :
	try :
		image = equalize(image)
		image, left_ear,  right_ear, left_pupil_ratio, right_pupil_ratio, left_eye_pos, right_eye_pos = detect_face(image)
		#image = cv2.resize(image, (0,0), fx = 0.5, fy= 0.5)
		#image = sharpen(image)
		#eyes = get_eyes(image)
		# if eyes is None :
		# 	return None
		# eyes = histogram_equalization(eyes.astype(np.uint8))
		# #eyes = get_eye(eyes)
		# #eyes = apply_thresholding(eyes)

		return image, left_ear, right_ear, left_pupil_ratio, right_pupil_ratio, left_eye_pos, right_eye_pos
	except Exception as e:
		print(e)
		return None, None, None, None, None, None, None

def main() :

	output_folder = "Preprocessed Output/"
	
	#create output folder if not exists
	if not os.path.exists(output_folder) :
		os.mkdir(output_folder)	

	for entity in os.listdir() :
		if entity.find("Images") != -1 :
			#for every image folder
			os.mkdir(output_folder + entity)

			#get the data file	
			file = open(entity + "/data.csv")
			lines = file.read()
			lines = lines.split("\n")[:-1]

			json_output = {
				"name" : [],
				"x_coord" : [],
				"y_coord" : [],
				"left_ear" : [],
				"right_ear" : [],
				"left_pupil" : [],
				"right_pupil" : [],
				"left_eye_pos" : [],
				"right_eye_pos" : []
			}

			#get every image from the data file
			count = 0
			errors = 0 
			for line in lines :
				line_tokens = line.split(",")
				image_name = line_tokens[0]
				image = cv2.imread(entity + "/" +image_name, cv2.IMREAD_COLOR)
				face, left_ear, right_ear, left_pupil_ratio, right_pupil_ratio, left_eye_pos, right_eye_pos = pipeline(image)
				if face is None :
					file.close()
					errors += 1
					continue
				#write image and append to new data file data
				#cv2.imwrite(output_folder + entity + "/" + image_name, face)
				json_output["name"].append(line_tokens[0])
				json_output["x_coord"].append(line_tokens[1])
				json_output["y_coord"].append(line_tokens[2])
				json_output["left_ear"].append(left_ear)
				json_output["right_ear"].append(right_ear)
				json_output["left_pupil"].append(left_pupil_ratio.tolist())
				json_output["right_pupil"].append(right_pupil_ratio.tolist())
				json_output["left_eye_pos"].append(left_eye_pos.tolist())
				json_output["right_eye_pos"].append(right_eye_pos.tolist())
				count += 1
				if count % 100 == 0 :
					print(count)
			file.close()

			#create output data file
			file = open(output_folder + entity + "/data.json", "+w")
			file.write(json.dumps(json_output))
			file.close()

			print(entity + "\nErrors" + str(errors) + "\n" + str(count))

if __name__ == "__main__" :
	main()
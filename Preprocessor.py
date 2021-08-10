import cv2
import os
import numpy as np
import shutil
from skimage.morphology import disk
from skimage.filters import rank

cv2_path = "/usr/local/lib/python3.8/dist-packages/cv2/data/"

def equalize(image) :
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(image)
	return cl1

def histogram_equalization(image) :
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	selem = disk(20)
	cl1 = rank.equalize(image, selem = selem)
	return cl1

def apply_thresholding(img, th_type = "OTSU") :
	th2 = None
	if th_type == "OTSU" :
		blur = cv2.GaussianBlur(img,(5,5),0)
		_,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	else :
		th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)	
	return th2

# def swap(arr, i, j) :
# 	temp = arr[i]
# 	arr[i] = arr[j]
# 	arr[j] = temp

# def sort_eyes(eyes) :
# 	for i in range(0, len(eyes)) :
# 		for j in range(i + 1, len(eyes)) :
# 			if eyes[i][0] < eyes[j][0] :
# 				swap(eyes, i, j)
# 	if len(eyes) >= 2 :
# 		swap(eyes, 1, len(eyes) - 1)
# 	return eyes

def detect_iris(image) :
	# Blur using 3 * 3 kernel. 
	#ray_blurred = cv2.blur(image, (3, 3)) 
  
	# Apply Hough transform on the blurred image. 
	image = cv2.medianBlur(image, 5)
	cv2.imshow("image", image)
	cv2.waitKey(0)
	detected_circles = cv2.HoughCircles(image,  
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 40, 
               param2 = 20, minRadius = 1, maxRadius = 40) 
	if detected_circles is None :
		return

	detected_circles = np.uint16(np.around(detected_circles)) 

	for pt in detected_circles[0, :] :
		a, b, r = pt[0], pt[1], pt[2]

		cv2.circle(image, (a, b), r, (0, 255, 0), 2) 
		cv2.circle(image, (a, b), 1, (0, 0, 255), 3) 

	cv2.imshow("image", image)
	cv2.waitKey(0)

def get_eyes(image) :
	'''
		Assume one face per image
		Params :
			image - input image
		Returns :
			eyes - list - list of the co-ordinates of both the eyes
	'''
	# eye_cascade = cv2.CascadeClassifier(cv2_path + "haarcascade_eye.xml")
	# #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	face_cascade = cv2.CascadeClassifier(cv2_path + "haarcascade_frontalface_default.xml")

	faces = face_cascade.detectMultiScale(image, 1.3, 5)

	# eyes_list = []

	for (x,y,w,h) in faces :
		face = image[y + (7 * h//20):y+ (h // 2), x + (w//6):x+ (5 * w // 6)]
		(fx, fy) = face.shape
		im1 = face[:,:fy//3]
		im2 = face[:,(2 * fy // 3):]
		face = cv2.hconcat([im1, im2])
		im1 = histogram_equalization(im1)
		detect_iris(im1)
		return face
		#cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		
		# eyes = eye_cascade.detectMultiScale(face)
		# eyes = sort_eyes(eyes)
		# for (ex, ey, ew, eh) in eyes:
		# 	eyes_list.append(face[ey:ey+eh, ex:ex+ew])
	#return eyes_list

def get_eye(image) :
	eye_cascade = cv2.CascadeClassifier(cv2_path + "haarcascade_eye.xml")
	eyes = eye_cascade.detectMultiScale(image, 1.3, 5)
	img_list = []
	for (x, y, w, h) in eyes :
		img = image[y:y+h, x:x+w]
		img_list.append(img)
	image = cv2.hconcat(img_list)
	return image

def sharpen(image) :
	gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
	unsharp_image = cv2.addWeighted(image, 2, gaussian_3, -1, 0, image)
	return unsharp_image

def pipeline(image) :
	image = equalize(image)
	#image = cv2.resize(image, (0,0), fx = 0.5, fy= 0.5)
	image = sharpen(image)
	eyes = get_eyes(image)
	if eyes is None :
		return None
	eyes = histogram_equalization(eyes.astype(np.uint8))
	#eyes = get_eye(eyes)
	eyes = apply_thresholding(eyes)
	return eyes

def main() :
	output_folder = "Preprocessed Output/"
	if not os.path.exists(output_folder) :
		#shutil.rmtree(output_folder)
		os.mkdir(output_folder)	
	for entity in os.listdir() :
		if entity.find("Images") != -1 :
			os.mkdir(output_folder + entity)	
			for image_name in os.listdir(entity) :
				if image_name.find(".csv") != -1: 
					file = open(entity + "/" + image_name)
					data = file.read()
					file.close()
					file = open(output_folder + entity + "/data.csv", "+w")
					file.write(data)
					file.close()
					continue
				image = cv2.imread(entity + "/" +image_name, cv2.IMREAD_COLOR)
				face = pipeline(image)
				if face is None :
					continue
				cv2.imwrite(output_folder + entity + "/" + image_name, face)

if __name__ == "__main__" :
	main()
from pynput.mouse import Listener, Button
import cv2
from datetime import datetime
import os
from queue import Queue
import threading, time
from threading import Lock
import pyautogui

# bufferless VideoCapture
base = "Images" + str(datetime.now().timestamp()) + "/"
os.mkdir(base)
file = open(base + "data.csv", "+w")


cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
for i in range(50) :
	cam.read()
print("Ready!!")


def main() :
	while True :
		ret, image = cam.read()
		if ret :
			filename = "image_" + str(datetime.now().timestamp()) + ".jpg"
			name = base + filename
			cv2.imwrite(name, image)
			x, y = pyautogui.position()
			file.write(filename+","+str(x)+","+str(y) + "\n")
			print(name)
		time.sleep(0.1)	

if __name__ == "__main__" :
	main()
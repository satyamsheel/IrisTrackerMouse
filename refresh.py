import os

for entry in os.listdir() :
	if entry.find("Images") != -1 :
		for files in os.listdir(entry) :
			if files.find("right_eye") != -1 or files.find("left_eye") != -1 or files.find("edge_image") != -1 :
				os.remove(entry + "/" + files)
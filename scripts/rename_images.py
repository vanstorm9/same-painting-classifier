from PIL import Image
import os
from time import time

starting_value = 0
testSelect = True


#root_path = '../scraped-images/no-vehicle/'
root_path = '../dataset/train/train/'
path_to_save = '../dataset/grayscale/little_train/'


if testSelect:
	#root_path = '../scraped-images/vehicle/'
	root_path = '../dataset/test/test/'
	path_to_save = '../dataset/grayscale/little_test/'


slash = '/'
root = os.listdir(root_path)

print 'Iterating through folders:'

t0 = time()


# Iterating through the item directories to get dir
for files in root:


	# To try to check if image
	try:
		img = Image.open(root_path + files).convert('L')
	except IOError:
		continue

	#width = img.size[0]
	#height = img.size[1]
	width = 64
	height = 64
	
	
	img2 = img.resize((width, height), Image.ANTIALIAS)

	removeStr = root_path + files
	saveStr = path_to_save + files

	#os.remove(removeStr)
	img2.save(saveStr)

	if starting_value%20 == 0:
		print starting_value

	
	starting_value = starting_value + 1

total_time = time() - t0
print 'Counted ', str(starting_value),' files'
print 'Resize time: ', total_time, 's'

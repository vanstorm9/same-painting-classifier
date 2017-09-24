import pandas as pd
import numpy as np
import os
#import cv2

#csvPath = '../csv/train_info.csv'
csvPath = '../csv/result.csv'

def getFilesFromDir(dirPath):

	imgPathAR = []	
	relPathAR = []

	for files in os.listdir(dirPath):

		fullStr = dirPath + files

		imgPathAR.append(fullStr)
		relPathAR.append(files)

	return imgPathAR, relPathAR

def convertPandaToNumpy(data):
	return df.as_matrix(data)



df = pd.read_csv(csvPath)

print df

#imgPathAR, relPathAR = getFilesFromDir(rootDataPath)

#targetIndex = df.loc[df['filename'] == relPathAR[0]]


'''
print 'Title'
print targetIndex['title'].values[0]
print 'Artist'
print targetIndex['artist'].values[0]
'''

#print convertPandaToNumpy(df)

'''
img = cv2.imread(imgPathAR[2])
cv2.imshow('window', img)
cv2.waitKey(0)
'''
'''
print ''
print ''
raw_input('. . .')
'''

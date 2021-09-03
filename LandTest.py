import cv2
import numpy as np
from imutils import contours
from skimage import measure
import imutils
import random as rng

#ImageInput(Copy the image destination in (path=r' ') as mentioned on your device, e.g. - C:\Users\AACC2\Desktop\Vegetation\test images\map.jpg)
path=r'C:\Users\Dell\Desktop\afforestattion\Vegetation\test images\3.png'
image=cv2.imread(path)
#cv2.imshow(" Initial image ", image)

#Converting RGB to HSV model
hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

#HSV range of colors to be detected
boundaries=[
    ([0,57,57],[70,255,255])]
'''#([20,100,100],[30,255,255]),
    #([25,36,54],[35,255,200]),
    #([17,15,100],[50,56,200]),
    #([86,31,4],[220,88,50]),
    #([25,146,190],[62,174,250]),
    #([103,86,65],[145,133,128])'''
    
#Loop to segregate unwanted color pixels from the hsv image
for (lower,upper) in boundaries:

    lower=np.array(lower,dtype = "uint8")
    upper=np.array(upper,dtype = "uint8")
    mask=cv2.inRange(hsv,lower,upper)
    output=cv2.bitwise_and(hsv,hsv,mask=mask)
    #cv2.imshow("images", np.hstack([hsv,output]))

#applying threshold to keep only those pixels who have a value of greater than 160, which is experimentally derived.
    ret, mask = cv2.threshold(output, 160, 255, cv2.THRESH_BINARY)

#applying functions to smoothen the processed image
    thresh = cv2.erode(mask, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

#converting the image from HSV to GrayScale
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('mask', thresh)
    

#Performing connected component analysis
    labels = measure.label(thresh, connectivity=2, background=0)
    #cv2.imshow('mask', labels)
    mask= np.zeros(thresh.shape, dtype="uint8")
    #cv2.imshow('thresh_image',mask)

#Loop to detect cluster of pixels which are nothing but our targetted area in an image
for label in np.unique(labels):
	if label == 0:
		continue

	labelMask = np.zeros(thresh.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv2.countNonZero(labelMask)
	if numPixels > 200:
		mask = cv2.add(mask, labelMask)

#using contours to map the boundary on the targetted area
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = contours.sort_contours(cnts)[0]

contours_poly = [None]*len(cnts)
boundRect = [None]*len(cnts)
centers = [None]*len(cnts)
radius = [None]*len(cnts)
for i, c in enumerate(cnts):
    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    boundRect[i] = cv2.boundingRect(contours_poly[i])
    centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

#drawing boundaries around the targetted area
for i in range(len(cnts)):
    #color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    color = int(16777215)
    cv2.drawContours(image, contours_poly, i, color,2)
    '''cv2.rectangle(image, (int(boundRect[i][0]), int(boundRect[i][1])), \
        (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
    cv2.circle(image, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)'''

#displaying the final processed image
cv2.imshow("Output Image(Picture)", image)
cv2.waitKey(0)

    
    #cv2.waitkey(0)



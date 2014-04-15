__author__ = 'brenemal'
import cv2
import numpy as np

def nothing(n):
    pass

"""
" See the comments in the color filter code since it is exactly the same code
"""
def getThresholdImage(frame, lower, upper):
    #convert frame to HSV to make it easier to work with
    frameImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #use your custom function to filter by color range
    #default :[110,50,50] [130,255,255]
    lowerRed = np.array(lower)
    upperRed = np.array(upper)
    imageThresh = cv2.inRange(frameImage, lowerRed, upperRed)
    #bitwise and adds the color back
    #imageThresh = cv2.bitwise_and(frameImage, frameImage, mask=imageThresh)
    return imageThresh


"""
" Pass this function a image that has been thresholded it will then
" overlay any contour boxes over the overlayImage on top of the repective areas
" the color is the color of the overlayed box the tracker is an array of previous
" entries for various colors
"""

def contourDetect(threshImage, overlayImage, color=(0,255,0), tracker=None):
    #make a copy so that the contour get doesnt mess stuff up
    countImage = threshImage.copy()
    #get the contours from the treshold image copy
    contours,hierarchy = cv2.findContours(countImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #foreach contour found draw a box around the area
    for cnt in contours:
        #get a simple bounding rect
        x,y,w,h = cv2.boundingRect(cnt)
        #get rid of really small boxes
        if cv2.contourArea(cnt) > 10000:
            cv2.rectangle(overlayImage,(x,y),(x+w,y+h),color,10)

def main():
    red_raw = cv2.imread("red.jpg")
    blue_raw = cv2.imread("blue.jpg")

    red_raw = cv2.resize(red_raw, (0,0), fx=0.25, fy=0.25)
    blue_raw = cv2.resize(blue_raw, (0,0), fx=0.25, fy=0.25)

    #generate the threshold images
    blueThresh = getThresholdImage(blue_raw, [100, 90, 30], [120, 255, 255])
    redThresh = getThresholdImage(red_raw, [120, 100, 60], [255, 255, 255])

     #generate the contour detection for every color
    contourDetect(blueThresh, blue_raw, (255, 0, 0))
    contourDetect(redThresh, red_raw, (0, 0, 255))

    cv2.imshow("Blue_raw", blue_raw)
    cv2.imshow("Blue", blueThresh)
    cv2.imshow("Red_raw", red_raw)
    cv2.imshow("Red", redThresh)
    cv2.waitKey(0)

main()
__author__ = 'brenemal'
import cv2
import numpy as np
import math

def nothing(n):
    pass

"""
" See the comments in the color filter code since it is exactly the same code
"""
def get_threshold_image(frame, lower, upper):
    #convert frame to HSV to make it easier to work with
    frame_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #use your custom function to filter by color range
    #default :[110,50,50] [130,255,255]
    lower = np.array(lower)
    upper = np.array(upper)
    image_thresh = cv2.inRange(frame_image, lower, upper)
    #bitwise and adds the color back
    #image_thresh = cv2.bitwise_and(frame_image, frame_image, mask=image_thresh)
    return image_thresh


"""
" Pass this function a image that has been thresholded it will then
" overlay any contour boxes over the overlayImage on top of the repective areas
" the color is the color of the overlayed box the tracker is an array of previous
" entries for various colors
"""
def contour_detect(thresh_image, overlay_image, color=(0,255,0), tracker=None):
    #make a copy so that the contour get doesnt mess stuff up
    count_image = thresh_image.copy()
    #get the contours from the treshold image copy
    contours, hierarchy = cv2.findContours(count_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #foreach contour found draw a box around the area
    area = 0
    rect = 0
    for cnt in contours:
        #get rid of really small boxes
        narea = cv2.contourArea(cnt)
        if narea    > area:
            rect = cv2.boundingRect(cnt)
            area = narea
    return rect


def main():
    #video = cv2.VideoCapture(1)
    video = cv2.VideoCapture("capture.avi")
    #grab a frame for calibration
    _, frame = video.read()
    while 1:
        _, frame = video.read()
        #generate the threshold images
        blue_thresh = get_threshold_image(frame,  [100, 150, 150], [120, 255, 255])
        red_thresh = get_threshold_image(frame, [0, 100, 60], [10, 255, 255])
        blue_raw = frame.copy()
         #generate the contour detection for every color
        blue = contour_detect(blue_thresh, blue_raw, (255, 0, 0))
        red = contour_detect(red_thresh, blue_raw, (0, 0, 255))
        print blue, red
        try:
            bx = blue[0] + (blue[2]/2)
            cv2.line(blue_raw, (bx, blue[1]), (bx, 0), (255, 0, 0), 5)
        except TypeError:
            pass
        try:
            rx = red[0] + (red[2]/2)
            cv2.line(blue_raw, (rx, red[1]), (rx, 0), (0, 0, 255), 5)
        except TypeError:
            pass
        cv2.imshow("detected", blue_raw)
        cv2.imshow("Blue", blue_thresh)
        cv2.imshow("Red", red_thresh)
        key = cv2.waitKey(0)
        if key == 113 or key == 1048689:
            break
        print key

main()
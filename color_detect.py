__author__ = 'brenemal'
import cv2
import numpy as np
import math

def nothing(n):
    pass

"""
" TODO List:
" Serial Comunication
" Scoring threshold
" Implement deeper stratagies
" Test on someone elses computer
"""


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
        if narea > area:
            rect = cv2.boundingRect(cnt)
            area = narea
    return rect


def find_arena_edge(frame):
    w, h, d = frame.shape
    start = w / 2
    initial = frame[start][-20]
    print "started with", initial
    left = right = 0
    for shift in range(1, w / 2):
        if left == 0:
            delta_left = int(initial[0]) - int(frame[start - shift][-20][0])
            if abs(delta_left) > 100:
                left = shift
                print "ended with ", frame[start - shift][-20]
        if right == 0:
            delta_right = int(initial[0]) - int(frame[start + shift][-20][0])
            if abs(delta_right) > 100:
                right = shift
                print "ended with ", initial, frame[start + shift][-20]
        initial = frame[start + shift][-20]
    print frame.shape
    #print (start+right, frame.shape[0]-20), (start-left, frame.shape[0]-20)
    cv2.line(frame, (start + right, frame.shape[0] - 20), (start - left, frame.shape[0] - 20), (255, 0, 0), 5)
    cv2.imshow("test", frame)


def thresh_delta(borders, field, y):
    dr, dl = 0, 0
    xsize = borders.shape[1] / 2
    for delta in range(xsize-1):
        if field[y][xsize + delta] != 0 and dr == 0:
            dr = delta
        if field[y][xsize - delta] != 0 and dl == 0:
            dl = delta

    return xsize - dl, xsize + dr


def extract_lines_old(borders):
    b = None
    start = 0
    for i in range(50):
        b = thresh_delta(borders, borders, i)
        if b[0] != 0:
            start = i
            break
    close = thresh_delta(borders, borders, 100)
    far = thresh_delta(borders, borders, 400)
    print "b", b
    print "close", close
    print "far", far
    return b, close, far, start

"""
" Takes in varibles about a pucks location and the field and
" initx: is the x of bottom left corner
" fl: the rate of change of f(y) = fl*x+initx on the left side
" fr: the same but the right side
" delta_x: the width of the base of the board in pixels
" projx, projy: the x and y you want to chnage to the real y
"""
def calc_real_y(initx, fl, fr, delta_x, projx, projy):
    right_pos = fr(projy)
    left_pos = fl(projy)
    delta = right_pos-left_pos
    #get the percent of the total board
    percent = float(projx-left_pos)/delta
    return percent*delta_x+initx


def draw_corners(im, corners):
    cv2.line(im, tuple(corners[0]), tuple(corners[1]), (0, 255, 255), 10)
    cv2.line(im, tuple(corners[1]), tuple(corners[3]), (0, 255, 0), 10)
    cv2.line(im, tuple(corners[3]), tuple(corners[2]), (255, 255, 0), 10)
    cv2.line(im, tuple(corners[2]), tuple(corners[0]), (100, 255, 100), 10)


def find_corners(frame):
    im = frame.copy()
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imgray = cv2.blur(imgray, (10, 10))
    thresh = cv2.Canny(imgray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(thresh, 1, math.pi / 180, 70, minLineLength=30, maxLineGap=50)
    borders = np.zeros(thresh.shape, np.uint8)
    max_dy = [[0, lines[0][0]], [0, lines[0][0]]]
    for line in lines:
        line = line[0]
        p1 = tuple(line[:2])
        p2 = tuple(line[2:])
        cv2.line(borders, p1, p2, (255, 0, 0), 5)
        dy = abs(p2[1] - p1[1])
        if dy > max_dy[0][0]:
            if abs(max_dy[0][1][1] - p1[1]) > 100:
                max_dy[1] = max_dy[0]
            max_dy[0] = [dy, line]
    cv2.imshow("borders", borders)
    cv2.waitKey(0)
    delta = [abs(max_dy[0][1][0] - max_dy[1][1][2]), abs(max_dy[0][1][2] - max_dy[1][1][0])]
    f_r = (delta[0] - delta[1]) / (max_dy[0][1][0] - max_dy[0][1][2])
    #top-left, top-right, bottom-left, bottom-right,
    corners = np.float32([
        [max_dy[0][1][2], max_dy[0][1][3]],
        [max_dy[0][1][2] + delta[1] + max_dy[0][1][3] * f_r, max_dy[0][1][3]],
        [max_dy[0][1][0], max_dy[0][1][1]],
        [max_dy[1][1][2], max_dy[1][1][3]],

    ])
    return corners, delta


def get_edge_functions(corners):
    divisor_r = (corners[3][0] - corners[1][0])
    if divisor_r == 0:
        divisor_r = .0001
    mr = (corners[3][1] - corners[1][1]) / divisor_r
    br = corners[3][1] - corners[3][0] * mr
    fr = lambda x: (x - br) / mr
    divisor_l = (corners[0][0] - corners[2][0])
    if divisor_l == 0:
        divisor_l = .001
    ml = (corners[0][1] - corners[2][1]) / divisor_l
    bl = corners[2][1] - corners[2][0] * ml
    fl = lambda y: (y - bl) / ml
    return fl, fr


def manual_corners(frame):
    im = frame.copy()
    #top-left, top-right, bottom-left, bottom-right
    corners = np.float32([
        [50, 50],
        [200, 50],
        [50, 500],
        [200, 500],

    ])
    cv2.namedWindow("settings")
    cv2.createTrackbar('top-left', 'settings', 50, im.shape[1], nothing)
    cv2.createTrackbar('top-right', 'settings', 200, im.shape[1], nothing)
    cv2.createTrackbar('top-y', 'settings', 50, im.shape[0], nothing)
    cv2.createTrackbar('bottom-left', 'settings', 50, im.shape[1], nothing)
    cv2.createTrackbar('bottom-right', 'settings', 200, im.shape[1], nothing)
    cv2.createTrackbar('bottom-y', 'settings', 500, im.shape[0], nothing)
    k = 0
    while k != 32:
        test = im.copy()
        corners = np.float32([
            [cv2.getTrackbarPos('top-left', 'settings'), cv2.getTrackbarPos('top-y', 'settings')],
            [cv2.getTrackbarPos('top-right', 'settings'), cv2.getTrackbarPos('top-y', 'settings')],
            [cv2.getTrackbarPos('bottom-left', 'settings'), cv2.getTrackbarPos('bottom-y','settings')],
            [cv2.getTrackbarPos('bottom-right', 'settings'), cv2.getTrackbarPos('bottom-y','settings')],
        ])
        draw_corners(test, corners)
        cv2.imshow("Calibrate", test)
        k = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow("Calibrate")
    cv2.destroyWindow("settings")
    delta = [abs(int(corners[2][0])-int(corners[3][0])), abs(int(corners[0][1])-int(corners[2 ][1]))]
    return corners, delta


def main():
    #video = cv2.VideoCapture(1)
    video = cv2.VideoCapture("capture.avi")
    #video = cv2.VideoCapture("test.avi")
    #grab a frame for calibration
    _, frame = video.read()
    #find_arena_edge(frame)
    corners, delta = find_corners(frame)
    im = frame.copy()
    draw_corners(im, corners)
    cv2.imshow("Press space to manually calibrate", im)
    k = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow("Press space to manually calibrate")
    if k == 32:
        corners, delta = manual_corners(frame)
    fl, fr = get_edge_functions(corners)

    while 1:
        _, frame = video.read()
        #generate the threshold images
        blue_thresh = get_threshold_image(frame,  [100, 150, 150], [120, 255, 255])
        red_thresh = get_threshold_image(frame, [0, 100, 60], [10, 255, 255])
        blue_raw = frame.copy()
         #generate the contour detection for every color
        blue = contour_detect(blue_thresh, blue_raw, (255, 0, 0))
        red = contour_detect(red_thresh, blue_raw, (0, 0, 255))
        pucks = {"red": None, "blue": None}
        try:
            bx = blue[0] + (blue[2]/2)
            base_x = calc_real_y(corners[2][0], fl, fr, delta[0], bx, blue[0])
            cv2.line(blue_raw, (bx, blue[1]), (int(base_x), int(corners[2][1])), (255, 0, 0), 5)
            pucks["blue"] = base_x
        except TypeError:
            pass
        try:
            rx = red[0] + (red[2]/2)
            base_x = calc_real_y(corners[2][0], fl, fr, delta[0], rx, red[0])
            cv2.line(blue_raw, (rx, red[1]), (int(base_x), int(corners[2][1])), (0, 0, 255), 5)
            pucks["red"] = base_x
        except TypeError:
            pass

        draw_corners(blue_raw, corners)
        cv2.imshow("detected", blue_raw)
        #cv2.imshow("Blue", blue_thresh)
        #cv2.imshow("Red", red_thresh)
        key = cv2.waitKey(0)  & 0xFF
        if key == 113 or key == 1048689:
            break
    cv2.destroyAllWindows()

main()
__author__ = 'brenemal'
import cv2
import numpy as np
import math
import time
import threading
import serial
import sys
from serial.tools import list_ports
from pyBusPirate.SPI import *

def nothing(n):
    pass


"""
" TODO List:
" Serial Comunication Testing
" Scoring threshold
" Implement deeper stratagies
" Test on someone elses computer
"""


class SerialHandler(object):
    baud = 9600
    timeout = 3
    port = ""
    ser = None

    def __init__(self, port, baud=9600, timeout=3):
        self.baud = baud
        self.timeout = timeout
        ser = serial.Serial(port, self.baud, timeout=self.timeout)
        print "Port Found:", port
        self.port = port
        self.ser = ser

    def list_serial_ports(self):
        for port in list_ports.comports():
            yield port[0]

    def write(self, char):
        self.ser.write(char)

    def read(self, count):
        return self.ser.read(count)

    def valid(self):
        return not self.ser is None


class SimpleSpi():
    def __init__(self, port="/dev/ttyUSB0"):
        self.port = port
        self.char = None
        self.spi = None

    def setup(self):
        self.spi = SPI("/dev/ttyUSB0", 115200)
        print "Entering binmode: ",
        if self.spi.BBmode():
            print "OK."
        else:
            print "failed."
            sys.exit()

        print "Entering raw SPI mode: ",
        if self.spi.enter_SPI():
            print "OK."
        else:
            print "failed."
            sys.exit()

        print "Configuring SPI."
        if not self.spi.cfg_pins(PinCfg.POWER):
            print "Failed to set SPI peripherals."
            sys.exit()
        if not self.spi.set_speed(SPISpeed._1MHZ):
            print "Failed to set SPI Speed."
            sys.exit()
        if not self.spi.cfg_spi(SPICfg.CLK_EDGE | SPICfg.OUT_TYPE):
            print "Failed to set SPI configuration."
            sys.exit()
        self.spi.timeout(0.2)
        self.transmit()

    def transmit(self):
        while True:
            if self.char is not None:
                print "start send"
                self.spi.CS_Low()
                #self.spi.bulk_trans(1, [selaf.char])
                self.spi.port.write(bytearray((0x10,)))
                self.spi.port.write(bytearray((self.char,)))
                self.spi.CS_High()
                print "sent", ord(self.char)
                self.char = None
            time.sleep(.001)


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
def contour_detect(thresh_image, overlay_image=None, color=(0, 255, 0), tracker=None):
    #make a copy so that the contour get doesnt mess stuff up
    count_image = thresh_image.copy()
    #get the contours from the treshold image copy
    _, contours, hierarchy = cv2.findContours(count_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


def rect_edge_detect(thresh_image, overlay_image, color=(0, 255, 0), tracker=None):
    #make a copy so that the contour get doesnt mess stuff up
    count_image = thresh_image.copy()
    #get the contours from the treshold image copy
    _, contours, hierarchy = cv2.findContours(count_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #foreach contour found draw a box around the area
    area = 0
    rect = 0
    lines = [[[0, 0], [0, 0]], [[700, 700], [700, 700]]]
    mid = thresh_image.shape[1]/2 + 30
    #an offset to be adjusted so that only rects far enough to the sides can be counted
    delta_mid = 60
    for cnt in contours:
        #get rid of really small boxes
        trect = cv2.boundingRect(cnt)
        cv2.rectangle(thresh_image, (trect[0], trect[1]), (trect[0]+trect[2],  trect[1]+trect[3]), (255), 20)
        print trect
        #if it is past the mid point we want the smallest x
        if trect[0] > mid+delta_mid and trect[1]+trect[3] > thresh_image.shape[0]-10:
            print trect, lines[1][1][0]
            if trect[0] < lines[1][1][0]:
                lines[1] = [[trect[0]+trect[2], trect[1]+trect[3]], [trect[0], trect[1]]]
        elif trect[0] < mid-delta_mid:
        #else we want the biggest
            if trect[0] > lines[0][0][0]:
                lines[0] = [[trect[0], trect[1]+trect[3]], [trect[0]+trect[2], trect[1]]]
    lines[0][0][0] += 10
    lines[0][1][0] += 0
    lines[1][0][0] -= 20
    lines[1][1][0] -= 0
    dy = [lines[0][0]+lines[0][1], lines[1][1]+lines[1][0]]
    return dy

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
    for delta in range(xsize - 1):
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
    delta = right_pos - left_pos
    #get the percent of the total board
    percent = float(projx - left_pos) / delta
    return percent * delta_x + initx, percent


def draw_corners(im, corners):
    cv2.line(im, tuple(corners[0]), tuple(corners[1]), (0, 255, 255), 2)
    cv2.line(im, tuple(corners[1]), tuple(corners[3]), (0, 255, 0), 2)
    cv2.line(im, tuple(corners[3]), tuple(corners[2]), (255, 255, 0), 2)
    cv2.line(im, tuple(corners[2]), tuple(corners[0]), (100, 255, 100), 2)


def draw_score_zones(im, score):
    cv2.line(im, tuple(score[0]), tuple(score[1]), (0, 255, 255), 2)
    cv2.line(im, tuple(score[2]), tuple(score[3]), (100, 255, 100), 2)


def find_corners(frame):
    im = frame.copy()
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imgray = cv2.blur(imgray, (10, 10))
    thresh = cv2.Canny(imgray, 100, 100, apertureSize=5)
    cv2.imshow("canny", thresh)
    lines = cv2.HoughLinesP(thresh, 1, math.pi / 180, 70, minLineLength=30, maxLineGap=50)
    borders = np.zeros(thresh.shape, np.uint8)
    max_dy = [[0, lines[0][0]], [0, lines[0][0]]]
    borders2 = borders.copy()
    mid = (borders.shape[1] / 2) + 30
    for line in lines:
        line = line[0]
        p1 = tuple(line[:2])
        p2 = tuple(line[2:])
        denom = (p1[1] - p2[1])
        x_diff = p2[0] - p1[0]
        y_diff = p2[1] - p1[1]
        angle = int(math.degrees(math.atan2(y_diff, x_diff)))

        cv2.line(borders2, p1, p2, (255, 0, 0), 5)
        if not ((70 < angle < 110) or (-70 > angle > -110)):
            continue
        print "got throug", angle
        cv2.line(borders, p1, p2, (255, 0, 0), 5)
        dy = abs(p2[1] - p1[1])
        if dy > max_dy[0][0] and p1[0] < mid:
            max_dy[0] = [dy, line]
        if dy > max_dy[1][0] and p1[0] > mid:
            max_dy[1] = [dy, line]
    #to undo rectangle based finding comment the next two lines
    rects = rect_edge_detect(borders, frame)
    max_dy = [[0, rects[0]], [0, rects[1]]]

    delta = [abs(max_dy[0][1][0] - max_dy[1][1][2]), abs(max_dy[0][1][2] - max_dy[1][1][0])]
    f_r = (delta[0] - delta[1]) / (max_dy[0][1][0] - max_dy[0][1][2])
    #top-left, top-right, bottom-left, bottom-right,
    corners = np.float32([
        [max_dy[0][1][2], max_dy[0][1][3]],
        [max_dy[0][1][2] + delta[1] + max_dy[0][1][3] * f_r, max_dy[0][1][3]],
        [max_dy[0][1][0], max_dy[0][1][1]],
        [max_dy[1][1][2], max_dy[1][1][3]],

    ])
    return corners, delta, lines


def find_score_zones(frame, lines, corners, fl, fr):
    borders = np.zeros(frame.shape, np.uint8)
    borders2 = borders.copy()
    for line in lines:
        line = line[0]
        p1 = tuple(line[:2])
        p2 = tuple(line[2:])
        denom = (p1[1] - p2[1])
        x_diff = p2[0] - p1[0]
        y_diff = p2[1] - p1[1]
        angle = int(math.degrees(math.atan2(y_diff, x_diff)))

        cv2.line(borders2, p1, p2, (255, 0, 0), 5)
        if not ((-10 < angle < 10) or (-190 < angle < -170) or (170 < angle < 190)):
            continue
        if not ((p1[0] < corners[0][0] < p2[0] and corners[0][1] < p1[1] < corners[2][1]) and
           (p1[0] < corners[1][0] < p2[0] and corners[1][1] < p1[1] < corners[3][1])):
            continue
        cv2.line(borders, p1, p2, (255, 0, 0), 5)
    max_dy = [[0, lines[0][0]], [0, lines[0][0]]]
    #cv2.imshow("borders", borders)
    #cv2.imshow("borders2", borders2)
    #cv2.waitKey(0)
    delta = [abs(max_dy[0][1][0] - max_dy[1][1][2]), abs(max_dy[0][1][2] - max_dy[1][1][0])]
    f_r = (delta[0] - delta[1]) / (max_dy[0][1][0] - max_dy[0][1][2])
    #top-left, top-right, bottom-left, bottom-right,
    corners = np.float32([
        [max_dy[0][1][2], max_dy[0][1][3]],
        [max_dy[0][1][2] + delta[1] + max_dy[0][1][3] * f_r, max_dy[0][1][3]],
        [max_dy[0][1][0], max_dy[0][1][1]],
        [max_dy[1][1][2], max_dy[1][1][3]],

    ])
    return corners, delta, lines


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


def score_zones(frame, fl, fr):
    im = frame.copy()
    #top-left, top-right, bottom-left, bottom-right
    score_zone = np.float32([
        [fl(10), 10],
        [fr(10), 10],
        [fl(200), 200],
        [fr(200), 200]
    ])
    cv2.namedWindow("settings")
    cv2.createTrackbar('top-y', 'settings', 50, im.shape[0], nothing)
    cv2.createTrackbar("delta-top", "settings", 150, 300, nothing)
    cv2.createTrackbar('bottom-y', 'settings', 500, im.shape[0], nothing)
    cv2.createTrackbar("delta-bottom", "settings", 150, 300, nothing)
    k = 0
    while k != 32:
        test = im.copy()
        top = cv2.getTrackbarPos('top-y', 'settings')
        dtop = cv2.getTrackbarPos('delta-top', 'settings')
        bottom = cv2.getTrackbarPos('bottom-y', 'settings')
        dbot = cv2.getTrackbarPos('delta-bottom', 'settings')
        score_zone = np.float32([
            [fl(top), top],
            [fr(top + (150 - dtop)), top + (150 - dtop)],
            [fl(bottom), bottom],
            [fr(bottom + (150 - dbot)), bottom + (150 - dbot)]
        ])
        draw_score_zones(test, score_zone)
        cv2.imshow("Calibrate", test)
        k = cv2.waitKey(1) & 0xFF
    st = (score_zone[1][1] - score_zone[0][1]) / (score_zone[1][0] - score_zone[0][0])
    score_top = lambda x, y: y < score_zone[0][1] + st * (x - score_zone[0][0])

    sb = (score_zone[3][1] - score_zone[2][1]) / (score_zone[3][0] - score_zone[2][0])
    score_bottom = lambda x, y: y > score_zone[2][1] + sb * (x - score_zone[2][0])

    cv2.destroyWindow("Calibrate")
    cv2.destroyWindow("settings")
    return score_zone, score_top, score_bottom


def crop_frame(frame):
    im = frame.copy()
    #top-left, top-right, bottom-left, bottom-right
    corners = np.float32([
        [50, 50],
        [200, 50],
        [50, 500],
        [200, 500],

    ])
    cv2.namedWindow("settings")
    cv2.createTrackbar('crop', 'settings', 50, im.shape[1], nothing)

    k = 0
    while k != 32:
        test = im.copy()
        y_crop = cv2.getTrackbarPos("crop", "settings")
        cv2.line(test, (0, y_crop), (test.shape[1], y_crop), (255, 0, 0))
        cv2.imshow("Calibrate", test)
        k = cv2.waitKey(1) & 0xFF
    cv2.destroyWindow("Calibrate")
    cv2.destroyWindow("settings")
    delta = [abs(int(corners[2][0]) - int(corners[3][0])), abs(int(corners[0][1]) - int(corners[2][1]))]
    return y_crop


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
            [cv2.getTrackbarPos('bottom-left', 'settings'), cv2.getTrackbarPos('bottom-y', 'settings')],
            [cv2.getTrackbarPos('bottom-right', 'settings'), cv2.getTrackbarPos('bottom-y', 'settings')],
        ])
        draw_corners(test, corners)
        cv2.imshow("Calibrate", test)
        k = cv2.waitKey(1) & 0xFF
    cv2.destroyWindow("Calibrate")
    cv2.destroyWindow("settings")
    delta = [abs(int(corners[2][0]) - int(corners[3][0])), abs(int(corners[0][1]) - int(corners[2][1]))]
    return corners, delta

def board_init(frame):
    #crop the other robot off the arena
    y_crop = crop_frame(frame)
    frame = frame[y_crop:, :]
    #find_arena_edge(frame)
    corners, delta, lines = find_corners(frame)
    fl, fr = get_edge_functions(corners)
    im = frame.copy()
    draw_corners(im, corners)
    cv2.imshow("Press space to manually calibrate", im)
    k = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow("Press space to manually calibrate")
    if k == 32:
        corners, delta = manual_corners(frame)
        im = frame
        draw_corners(im, corners)
        fl, fr = get_edge_functions(corners)
    score_zone, score_top, score_bottom = score_zones(im, fl, fr)
    return corners, delta, fl, fr, frame, score_zone, y_crop


spi = None
USESERIAL = False
def main():
    #video = cv2.VideoCapture(1)
    #video = cv2.VideoCapture("capture.avi")
    #video = cv2.VideoCapture("test.avi")
    video = cv2.VideoCapture("test_real.avi")
    #grab a frame for calibration
    _, frame = video.read()

    if USESERIAL:
        #initalize the pirate bus in a sperate thread
        spi = SimpleSpi("/dev/ttyUSB0")
        t = threading.Thread(target=spi.setup)
        t.start()
    else:
        print "No serial device detected. Starting test mode."

    corners, delta, fl, fr, frame, score_zone, y_crop = board_init(frame)

    #create a mask to remove background noise
    mask = np.zeros(frame.shape[:2], np.uint8)
    top = min(corners[0][1], corners[1][1])
    bot = max(corners[2][1], corners[3][1])
    left = min(corners[0][0], corners[2][0])
    right = max(corners[1][0], corners[3][0])
    for i in range(mask.shape[0]):
        mask[i, fl(i):fr(i)] = 255
    #keep track of the previous point to avoid constant adjusting
    prev = 0
    while 1:
        _, frame = video.read()
        #crop the frame
        frame = frame[y_crop:, :]
        frame = cv2.bitwise_and(frame, frame, mask=mask)

        #generate the threshold images based on color
        blue_thresh = get_threshold_image(frame, [100, 150, 150], [120, 255, 255])
        red_thresh = get_threshold_image(frame, [0, 100, 60], [10, 255, 255])

        #generate the contour detection for every color
        blue = contour_detect(blue_thresh, frame, (255, 0, 0))
        red = contour_detect(red_thresh, frame, (0, 0, 255))

        # use this dict to make sure that we can keep track of which pucks are valid targets, this allows for better
        # strategies like marking pucks that have been scored, without a ton of messy logic
        pucks = {"red": None, "blue": None}
        blue_percent, red_percent = -1, -1
        try:
            #get the x position of the center of the blue puck
            bx = blue[0] + (blue[2] / 2)
            #get the base_x and percentage the blue puck is along the x axis
            base_x, blue_percent = calc_real_y(corners[2][0], fl, fr, delta[0], bx, blue[1])
            #draw the line for visualization
            cv2.line(frame, (bx, blue[1]), (int(base_x), int(corners[2][1])), (255, 0, 0), 5)
            pucks["blue"] = base_x
        except TypeError:
            pass
        try:
            rx = red[0] + (red[2] / 2)
            base_x, red_percent = calc_real_y(corners[2][0], fl, fr, delta[0], rx, red[1])
            cv2.line(frame, (rx, red[1]), (int(base_x), int(corners[2][1])), (0, 0, 255), 5)
            pucks["red"] = base_x
        except TypeError:
            pass
        size = 0x9a
        red_percent = -1
        if red_percent != -1:
            #print "red:", int(round(red_percent * 255, 0))
            char = size-int(round(red_percent * size, 0))
            red_char = char
            cv2.putText(frame, str(red_char), (rx, red[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0))
            #print char
            if 0 < char < 256 and abs(prev-char) > 2 and USESERIAL:
                spi.char = chr(char)
                prev = char
                pass

            #ser.write(chr(int(round(red_percent * 255, 0))))
        elif blue_percent != -1:
            #print "red:", int(round(blue_percent * 255, 0))
            char = size-int(round(blue_percent * size, 0))
            blue_char = char
            cv2.putText(frame, str(blue_char), (bx, blue[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0))
            #print char
            if 0 < char < 256 and abs(prev-char) > 2 and USESERIAL:
                spi.char = chr(char)
                prev = char
                pass
        else:
            print "no valid target"
        draw_corners(frame, corners)
        draw_score_zones(frame, score_zone)
        cv2.imshow("detected", frame)
        #cv2.imshow("Blue", blue_thresh)
        #cv2.imshow("Red", red_thresh)

        key = cv2.waitKey(1) & 0xFF
        if key == 113 or key == 1048689:
            cv2.destroyAllWindows()
            sys.exit(1)
            break
    cv2.destroyAllWindows()


main()
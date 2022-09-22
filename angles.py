import cv2
import numpy as np
from PIL import Image
import tracker
from mss import mss
from itertools import combinations
import math


# WIDTH, HEIGHT = 640, 480
WIDTH, HEIGHT = 1280, 720 
TOP, LEFT = 455, 80
RADIUS = 250

tracker = tracker.EuclideanDistTracker()


class MSSSource:
    def __init__(self):
        self.sct = mss()

    def frame(self):
        monitor = {'top': TOP, 'left': LEFT, 'width': WIDTH, 'height': HEIGHT}
        im = np.array(self.sct.grab(monitor))
        im = np.flip(im[:, :, :3], 2)  # 1
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 2
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return True, im

    def release(self):
        pass

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)

    if cam.isOpened(): 
        # get vcap property 
        WIDTH  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
        HEIGHT = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

    source = MSSSource()
    while (True):
        ret, img = cam.read()
        # img = cv2.flip(img,1)
        # ret, img = source.frame()
        # img = cv2.imread('orange.jpg')


        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([100,100,90]), np.array([200,255,255]))
        
        # _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
       
        detections = []
        for cnt in contours:
        #     # Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if area > 100:
                # cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(cnt)
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                detections.append([x, y, w, h])
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x1, y1, x2, y2, midx, midy, z, id = box_id
            # cv2.putText(img, str(id), (x1, y1 - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            # cv2.rectangle(img, (x1, y1), (x1 + x2, y1 + y2), (0, 255, 0), 3)
            # stats = tracker.history[id]

        no_objects = len(boxes_ids)
        if len(boxes_ids) == 2:
            for a,i in combinations(boxes_ids, 2):
                cv2.line(img, (int(i[0]), int(i[1])), (int(a[0]), int(a[1])), (0,255,0), 2)

        elif len(boxes_ids) == 1:
            # print((boxes_ids[0][0], boxes_ids[0][1]))
            # cv2.circle(img, (boxes_ids[0][0], boxes_ids[0][1]), 2, (0,0,255), -1)
            cv2.line(img, (0, int(HEIGHT/2)), (WIDTH, int(HEIGHT/2)), (0,0,255), 1)
            cv2.line(img, (int(WIDTH/2), 0), (int(WIDTH/2), HEIGHT), (0,0,255), 1)
            cv2.circle(img, (int(WIDTH/2), int(HEIGHT/2)), RADIUS, (0,0,255), 3)


            opposite = (boxes_ids[0][0], int(HEIGHT/2)), (boxes_ids[0][0], boxes_ids[0][1])
            opposite_len = (opposite[0][1] - opposite[1][1])
            
            adjacent = (int(WIDTH/2), int(HEIGHT/2)), (boxes_ids[0][0], int(HEIGHT/2))
            adjacent_len = (adjacent[0][0] - adjacent[1][0])/1

            hypotonuse = (int(WIDTH/2), int(HEIGHT/2)), (boxes_ids[0][0], boxes_ids[0][1])
            hypotonuse_len = math.sqrt((opposite_len**2 + adjacent_len**2))

            cv2.line(img, hypotonuse[0], hypotonuse[1], (0,255,0), 3) # HYPOTONUSE
            cv2.line(img, opposite[0], opposite[1], (0,255,0), 3) # OPPOSITE
            cv2.line(img, adjacent[0], adjacent[1], (0,255,0), 3) # ADJACENT

            if hypotonuse != 0: 
                ratio = round((adjacent_len/hypotonuse_len), 5)
                angle = (math.acos(adjacent_len/hypotonuse_len)*180)/math.pi
                # angle = (math.asin(ratio)*180)/math.pi

                polar_x = RADIUS*math.cos(angle)*(math.pi/180)
                polar_y = RADIUS*math.sin(angle)*(math.pi/180)
                cv2.putText(img, str(round(angle, 2)), (int(WIDTH/2), int(HEIGHT/2)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                # cv2.line(img, (int(WIDTH/2), int(HEIGHT/2)), (int(polar_x), int(polar_y)), (0,255,0), 3) # HYPOTONUSE

        elif len(boxes_ids) == 3:
            triangle_cnt = np.array( [[i[0], i[1]] for i in boxes_ids] )
            cv2.drawContours(img, [triangle_cnt], 0, (0,255,0), -1)

        cv2.imshow('my webcam', img)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    source.release()
    cv2.destroyAllWindows()

# make a mask for colour orange

# assign the masked objects an ID

# draw lines between centroids of the objects

# calculate the angles between the lines

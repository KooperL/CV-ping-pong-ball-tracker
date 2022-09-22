import cv2
import numpy as np
from PIL import Image
from mss import mss
from itertools import combinations
import math

import tracker

# WIDTH, HEIGHT = 640, 480
WIDTH, HEIGHT = 1280, 720 
TOP, LEFT = 455, 80
RADIUS = 250

tracker = tracker.EuclideanDistTracker()

velocity_history = {}

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
        img = cv2.flip(img,1)
        # ret, img = source.frame()
        # img = cv2.imread('orange.jpg')


        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([100,150,180]), np.array([110,255,255]))
        
        # _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
       
        detections = []
        for cnt in contours:
        #     # Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if area > 300:
                # cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(cnt)
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                detections.append([x, y, w, h])
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x1, y1, w, h, cx, cy, z, id = box_id
            x2 = x1 + w
            y2 = y1 + h
            # cv2.putText(img, str(z), (x1, y1 - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(img, str(id), (x1, y1-h), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            # cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 3)
            stats = tracker.history[id]

            x1, y1, w, h, midx, midy, z, id = box_id
            x2 = x1 + w
            y2 = y1 + h
            try:
                print(stats)
                time_difference = (stats[0][1] - stats[len(stats)-1][1]).total_seconds()
                distance = np.sqrt(np.sum((stats[0][0] - stats[len(stats)-1][0])**2))   # FAILS IF OFF SCREEN
                direction = stats[0][0] - stats[len(stats)-1][0]
                if time_difference != 0 and distance != 0:
                    speed = distance/time_difference
                    velocity = (speed/distance)*direction
                    displacement = velocity * 1 # time
                    aim_here = (displacement + stats[0][0])# + (gravity*time)
                    aim_here = [int(i) for i in aim_here]

                    if id in velocity_history:
                        velocity_history[id].insert(0, aim_here)
                        if len(velocity_history[id]) > 10:
                            velocity_history[id].pop()
                    else:
                        velocity_history[id] = [aim_here]

                    # cv2.line(img, (int(aim_here[0]), int(aim_here[1])), (midx, midy), (0,0,255), 1)
                    if len(velocity_history[id]) > 1:
                        smooth_aim_here = np.mean(velocity_history[id], axis=0)
                        cv2.line(img, (int(smooth_aim_here[0]), int(smooth_aim_here[1])), (midx, midy), (0,0,255), 1)

                        ## Parabola time
                        # pts = np.array([[25, 70], [25, 160], 
                        #     [110, 200], [200, 160], 
                        #     [200, 70], [110, 20]],
                        #    np.int32)
                        # pts = np.array([[i, (-i**2)+(i*smooth_aim_here[1])] for i in range(WIDTH-midx)])
                        # print(pts)
                        # pts = pts.reshape((-1, 1, 2))
                        # cv2.polylines(img, [pts], False, (255,0,0), 2)



                # tracker.history[object_id] = objects_bbs_ids
                # apx_distance = round((y2-y1), 2)
                cv2.putText(img, f'D: {z}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)
                # cv2.line(img, (int(WIDTH/2), int(HEIGHT/2)), (int(midx), int(midy)), (150,150,150), 1)
            except ValueError:
                pass

        cv2.imshow('my webcam', img)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    source.release()
    cv2.destroyAllWindows()

# make a mask for colour orange

# assign the masked objects an ID

# draw lines between centroids of the objects

# calculate the angles between the lines

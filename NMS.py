import numpy as np
import cv2


nms_window = 5
length = len(cornerList)
for i in range(length-nms_window):
    if cornerList[i][2]>0:
        for j in range(1, nms_window+1):
            if (abs(cornerList[i][0]-cornerList[i+j][0]) <=nms_window) and (abs(cornerList[i][1]-cornerList[i+j][1])<= nms_window):
                cornerList[i+j][2]=0


for r in cornerList:
    if r[2]==0:
        cornerList.remove(r)
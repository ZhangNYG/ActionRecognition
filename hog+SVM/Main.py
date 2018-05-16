import cv2
import numpy as np
from Detection import *
from Tracker import *


if __name__ == '__main__':
    vc = cv2.VideoCapture("../video.avi")
    index = 0

    dt = Detector()
    tr = Tracker()





    while(True):
        ret,frame = vc.read()
        cands, imgs, frame = dt.detectCandidates(frame)
        tr.updateTracks(cands,imgs,frame,index)
        tr.drawTracks(frame)

        # cv2.imshow("salam",frame)

        if(cv2.waitKey(0)==27):
            cv2.imwrite("detection.png",frame)
            break


        # show the output images
        # if(index%5==0):
        #     print index
        #     cv2.imwrite("detection.png",frame)
        #     cv2.waitKey(0)

        index += 1




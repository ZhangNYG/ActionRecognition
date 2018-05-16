import cv2
import numpy as np
from Detection import *
from Tracker import *


# video_path = "video.avi"
# video_path = "sample_walking.avi"
input_video = "demo.mp4"

output_video = 'outsample.avi'
Save_video = False


if __name__ == '__main__':
    vc = cv2.VideoCapture(input_video)
    index = 0

    dt = Detector(show=True)
    tr = Tracker()
    #Saving video
    if(Save_video):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video,fourcc, 20.0, (400, 400))

    while(cv2.waitKey(5)!=27):
        ret,frame = vc.read()
        if not ret:
            break
        if(index%2==0 and index>=0):
            frame = cv2.resize(frame, (400, 400))
            cands, feat_list, frame = dt.detectCandidates(frame)
            tr.updateTracks(cands, feat_list, frame, index)
            frame = tr.drawTracks()

            if(Save_video):
                out.write(frame)



        index += 1

    vc.release()
    
    if(Save_video):
        out.release()

    cv2.destroyAllWindows()


import cv2
import numpy as np
from Detection import *
from Tracker import *
import argparse







if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--in", type=str, default="demo.mp4",
	                    help="Input Video")
	parser.add_argument("--out", type=str, default=None,
	                    help="Output Video")

	args = parser.parse_args()
	input_video = args.in
	Output = args.out

	if(output_video is None):
	    Save_video = False
	else:
	    Save_video = True

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


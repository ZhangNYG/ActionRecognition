import cv2
import numpy as np
from Detection import *
from Saver import Tracker
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_data_dir", type=str, default="VideoDataset/",
                        help="Video Dataset folder")

    args = parser.parse_args()
    directory = args.video_data_dir

    with open("processed_videos.txt") as f:
        traversed = [x.strip() for x in f.readlines()]


    with open("checkpoint.txt") as f:
        video, idx = f.readline().split(" ")


    idx = int(idx)

    dt = Detector()
    for root, dirs, files in os.walk(directory):
        for file in files:
            videopath = os.path.join(root, file)
            if file.endswith('.avi'):

                if (videopath in traversed):
                    continue


                tr = Tracker()
                vc = cv2.VideoCapture(videopath)
                index = 0

                while(vc.isOpened()and cv2.waitKey(1) != 27):
                    ret,frame = vc.read()

                    if(videopath == video):
                        while(index<=idx-30):
                            ret, frame = vc.read()
                            index +=1
                    if ret:
                        if(index%3==0):
                            cands, feat_list, frame = dt.detectCandidates(frame)
                            tr.updateTracks(cands,feat_list,frame,index)
                            tr.drawTracks()
                            tr.saveAction(root)
                    else:
                        break

                    file = open("checkpoint.txt", mode='w+')
                    file.write(videopath + " " + str(index))
                    file.close()
                    index += 1

                file = open("processed_videos.txt", mode='a+')
                file.write(videopath + "\n")
                file.close()

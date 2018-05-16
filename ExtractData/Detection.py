import cv2
import imutils
import numpy as np
import sys
sys.path.insert(0, '../pose_estimation/')


from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

needed_elements = np.arange(0,14,1)
class Detector:
    def __init__(self):
        model = 'mobilenet_thin'
        w, h = model_wh('432x368')
        self.estimator = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    def detectCandidates(self,frame):

        cands = []
        humans = self.estimator.inference(frame)
        image_h, image_w = frame.shape[:2]
        frame = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)
        feat_list = []
        for i in range(len(humans)):
            if i>=len(humans):
                break

            keys = humans[i].body_parts.keys()
            if len(np.setdiff1d(needed_elements, keys)):
                del humans[i]
                continue
            neck = humans[i].body_parts[1]
            lhip = humans[i].body_parts[8]
            rhip = humans[i].body_parts[11]
            center =  (neck.x + lhip.x + rhip.x) / 3, (neck.y + lhip.y + rhip.y)/3



            feats = []
            for idx in needed_elements:
                part = humans[i].body_parts[idx]
                feats = feats + [part.x-center[0],part.y-center[1]]

            feat_list.append(np.asarray(feats))

            center = image_w*center[0],image_h*center[1]
            cv2.circle(frame,(int(center[0]),int(center[1])),3,(255,0,0),3)
            cands.append(np.asarray(center,dtype=np.float32))


        # print feat_list[0]


        return cands,feat_list,frame
            # show some information on the number of bounding boxes



import argparse
import time

import cv2
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh



fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')

    args = parser.parse_args()
    model = 'mobilenet_thin'
    w, h = model_wh('432x368')
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)

    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret_val, image = cap.read()


        humans = e.inference(image)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break


    cv2.destroyAllWindows()

import cv2
import imutils
import numpy as np


class Detector:
    def __init__(self,stride = 8, padding = 36, scale = 1.1, min_width=400):
        # self.subtractor = cv2.createBackgroundSubtractorMOG2(history=history)
        # self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.stride = (stride,stride)
        self.padding = (padding,padding)
        self.scale = scale
        self.width = min_width

    def detectCandidates(self,frame):

        # # print self.subtractor
        # # mask = self.subtractor.apply(frame)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        # print mask.shape
        # cv2.imshow('mask',mask)
        frame = imutils.resize(frame, width=min(self.width, frame.shape[1]))
        # detect people in the image
        (rects, weights) = self.hog.detectMultiScale(frame, winStride=self.stride,
                                                padding=self.padding, scale=self.scale)

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        # pick = non_max_suppression(rects, probs=None, overlapThresh=0.55)

        pick = nms(rects,overlapdist_th=0.55)
        # draw the final bounding boxes
        cands = []
        imgs = []
        for (xA, yA, xB, yB) in pick:
            # cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            cands.append(np.array([(xA + xB) / 2, (yA + yB) / 2]))
            imgs.append(frame[yA:yB,xA:xB,:])



        return cands,imgs,frame
            # show some information on the number of bounding boxes


# Our implementation of Non-max Suppression
def nms(boxes, overlapdist_th):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapdist_th)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

# Exponential weighted average
# Function will return average velocity over last frame_n

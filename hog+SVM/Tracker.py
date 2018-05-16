from scipy.spatial import distance_matrix
import numpy as np
import cv2

class Track:
    def __init__(self, point, lastFrame, img,predicted = 0):
        self.points = []
        self.points.append((point,lastFrame,img,predicted))

    def getBox(self):
        return self.box

    def setBox(self,box):
        self.box = box

    def getLastIdx(self):
        return self.points[-1][1]

    def isLastPredicted(self):
        return self.points[-1][3]

    def getLastImage(self):
        return self.getImgSeqs()[-1]

    def getImgSeqs(self):
        return [point[2] for point in self.points]

    def getRect(self):
        point = np.int32(self.getLastPoint())
        h,w,_ = self.getLastImage().shape
        return point[0]-w/2,point[1]-h/2,point[0]+w/2,point[1]+h/2

    def getTrack(self):
        return [point[0] for point in self.points]

    def getIdxs(self):
        return [point[1] for point in self.points]

    def getLastPoint(self):
        return self.points[-1][0]

    def getLen(self):
        return len(self.points)

    def addPoint(self,point,last_idx,img,predicted = 0):
        self.points.append((point,last_idx,img,predicted))

    def delFirst(self):
        del self.points[0]


class Tracker:
    def __init__(self,len_lim=10,dist_thresh=25,frame_th = 5):
        self.tracks = []
        self.len_lim = len_lim
        self.dist_th = dist_thresh
        self.frame_th = frame_th
        self.frame_idx = 0
        self.frame = None

    def updateTracks(self,cands,imgs,frame,frame_idx):
        self.frame_idx = frame_idx
        self.frame = frame
        if len(cands) == 0:
            return

        if len(self.tracks) == 0:
            self.tracks = [Track(cand, frame_idx, img) for cand,img in zip(cands,imgs)]
            return

        last_points = [track.getLastPoint() for track in self.tracks]
        dist_toCands = distance_matrix(last_points, cands)
        idx_min_toCands = np.argmin(dist_toCands, axis=1)
        min_toTracks = np.min(dist_toCands, axis=0)
        n_del = 0
        i = 0
        # list of the candidate regions which became tracks
        trackparts = []
        while (i < len(self.tracks)):
            # j is the index of track in the
            # distance matrix
            # i is the current track index in the list of tracks
            j = n_del + i

            cand_idx = idx_min_toCands[j]
            min_toCand = dist_toCands[j, cand_idx]
            if frame_idx - self.tracks[i].getLastIdx() > self.frame_th:
                del self.tracks[i]
                n_del += 1
                continue

            if (min_toCand < self.dist_th and min_toCand <= min_toTracks[cand_idx]):
                self.tracks[i].addPoint(cands[cand_idx], frame_idx, imgs[cand_idx])
                trackparts.append(cand_idx)
            else:
                self.addPredPoint(i)
            i += 1

        remaining_idxs = [cand for cand in range(len(cands)) if cand not in trackparts]

        for i in remaining_idxs:
            # print imgs[i].shape
            track = Track(cands[i], frame_idx, imgs[i])
            self.tracks.append(track)
            continue

        return

    def drawTracks(self,frame):
        cv2.polylines(frame, [np.int32(tr.getTrack()) for tr in self.tracks], False, (0, 255, 0))
        for track in self.tracks:
            xA,yA,xB,yB = track.getRect()
            if track.getLastIdx() == self.frame_idx:
                color = (0, 255, 0)
                if track.isLastPredicted():
                    color = (255, 0, 0)
                    # print track.isLastPredicted
                cv2.rectangle(frame, (xA, yA), (xB, yB),color , 2)
        cv2.imshow("After NMS", frame)
        return

    def saveSequences(self):
        for track in self.tracks:
            if track.getLen() >= 5:
                pass


    def addPredPoint(self,i):
        vels = calcVels(self.tracks[i])
        if (len(vels) > 0):
            # print vels
            v_avg = ewma(vels)
            last_coord = self.tracks[i].getLastPoint()
            new_pos = predpos(v_avg, last_coord,
                              self.tracks[i].getLastIdx(), self.frame_idx)
            if np.linalg.norm(last_coord - new_pos) < self.dist_th:
                xA, yA, xB, yB = self.tracks[i].getRect()
                img = self.frame[yA:yB,xA:xB,:]
                self.tracks[i].addPoint(new_pos, self.frame_idx, img, predicted=1)


def calcVels(track):
    pos = track.getTrack()
    time = track.getIdxs()
    vels = []
    for i in range(len(pos)-1):
        vels.append((pos[i+1]-pos[i])/(time[i+1]-time[i]))
    return vels

def ewma(vels, frame_n=3):
    beta = 1 - 1.0/frame_n
    v_avg = 0
    for v in vels:
        v_avg = beta * v_avg + (1 - beta) * v
    return v_avg

def predpos(v_avg,point,last_frame,current_frame):
    pos = point + v_avg*(current_frame-last_frame)
    return pos


"""

# # Python 2/3 compatibility
# import sys
# PY3 = sys.version_info[0] == 3
# 
# if PY3:
#     long = int
# 
# import cv2 as cv
# from math import cos, sin, sqrt
# import numpy as np
# 
# if __name__ == "__main__":
# 
#     img_height = 500
#     img_width = 500
#     kalman = cv.KalmanFilter(2, 1, 0)
# 
#     code = long(-1)
# 
#     cv.namedWindow("Kalman")
# 
#     while True:
#         state = 0.1 * np.random.randn(2, 1)
# 
#         kalman.transitionMatrix = np.array([[1., 1.], [0., 1.]])
#         kalman.measurementMatrix = 1. * np.ones((1, 2))
#         kalman.processNoiseCov = 1e-5 * np.eye(2)
#         kalman.measurementNoiseCov = 1e-1 * np.ones((1, 1))
#         kalman.errorCovPost = 1. * np.ones((2, 2))
#         kalman.statePost = 0.1 * np.random.randn(2, 1)
# 
#         while True:
#             def calc_point(angle):
#                 return (np.around(img_width/2 + img_width/3*cos(angle), 0).astype(int),
#                         np.around(img_height/2 - img_width/3*sin(angle), 1).astype(int))
# 
#             state_angle = state[0, 0]
#             state_pt = calc_point(state_angle)
# 
#             prediction = kalman.predict()
#             predict_angle = prediction[0, 0]
#             predict_pt = calc_point(predict_angle)
# 
#             measurement = kalman.measurementNoiseCov * np.random.randn(1, 1)
# 
#             # generate measurement
#             measurement = np.dot(kalman.measurementMatrix, state) + measurement
# 
#             measurement_angle = measurement[0, 0]
#             measurement_pt = calc_point(measurement_angle)
# 
#             # plot points
#             def draw_cross(center, color, d):
#                 cv.line(img,
#                          (center[0] - d, center[1] - d), (center[0] + d, center[1] + d),
#                          color, 1, cv.LINE_AA, 0)
#                 cv.line(img,
#                          (center[0] + d, center[1] - d), (center[0] - d, center[1] + d),
#                          color, 1, cv.LINE_AA, 0)
# 
#             img = np.zeros((img_height, img_width, 3), np.uint8)
#             draw_cross(np.int32(state_pt), (255, 255, 255), 3)
#             draw_cross(np.int32(measurement_pt), (0, 0, 255), 3)
#             draw_cross(np.int32(predict_pt), (0, 255, 0), 3)
# 
#             cv.line(img, state_pt, measurement_pt, (0, 0, 255), 3, cv.LINE_AA, 0)
#             cv.line(img, state_pt, predict_pt, (0, 255, 255), 3, cv.LINE_AA, 0)
# 
#             kalman.correct(measurement)
# 
#             process_noise = sqrt(kalman.processNoiseCov[0,0]) * np.random.randn(2, 1)
#             state = np.dot(kalman.transitionMatrix, state) + process_noise
# 
#             cv.imshow("Kalman", img)
# 
#             code = cv.waitKey(100)
#             if code != -1:
#                 break
# 
#         if code in [27, ord('q'), ord('Q')]:
#             break
# 
#     cv.destroyWindow("Kalman")
"""
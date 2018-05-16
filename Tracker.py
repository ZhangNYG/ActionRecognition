from scipy.spatial import distance_matrix
import numpy as np
import cv2
import xgboost as xgb
import pickle


class Track:
    # feats stands for features for one specific human in the current frame
    # feat_list stands for list of features of all applicable humans in the current frame

    def __init__(self, point, frame_n, feat):
        self.points = []
        self.points.append((point, frame_n, feat))
        self.act_idx = -1

    def getLastIdx(self):
        return self.points[-1][1]

    def getDataSample(self):
        features = np.zeros(0, dtype=np.float32)
        start_frame = self.points[0][1]
        for point in self.points:
            features = np.hstack((features, point[1] - start_frame))
            features = np.hstack((features, point[2]))
        return features[1:]

    def setActionIdx(self, idx):
        self.act_idx = idx

    def getActionIdx(self):
        return self.act_idx

    def getTrack(self):
        return [point[0] for point in self.points]

    def getIdxs(self):
        return [point[1] for point in self.points]

    def getLastPoint(self):
        return self.points[-1][0]

    def getLen(self):
        return len(self.points)

    def addPoint(self, point, last_idx, feat):
        self.points.append((point, last_idx, feat))

    def delFirst(self):
        del self.points[0]


class Tracker:
    def __init__(self, len_lim=10, dist_thresh=50, frame_th=5):
        self.tracks = []
        self.len_lim = len_lim
        self.dist_th = dist_thresh
        self.frame_th = frame_th
        self.frame_idx = 0
        self.frame = None
        self.last_act_idx = 0
        self.clf = pickle.load(open("TrainActivityRecognition/clf.dat", "rb"))
        self.labels = pickle.load(open("TrainActivityRecognition/labels.dat", "rb"))

    def updateTracks(self, cands, feat_list, frame, frame_idx):
        self.frame_idx = frame_idx
        self.frame = frame
        if len(cands) == 0:
            return

        if len(self.tracks) == 0:
            self.tracks = [Track(cand, frame_idx, feat) for cand, feat in zip(cands, feat_list)]
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
                self.tracks[i].addPoint(cands[cand_idx], frame_idx, feat_list[cand_idx])
                trackparts.append(cand_idx)

            i += 1

        remaining_idxs = [cand for cand in range(len(cands)) if cand not in trackparts]

        for i in remaining_idxs:
            # print feat_list[i].shape
            track = Track(cands[i], frame_idx, feat_list[i])
            self.tracks.append(track)

        return

    def predActions(self):
        for track in self.tracks:
            if track.getLen() >= 10:
                p = track.getLastPoint().astype(np.int32)
                track.setActionIdx(self.last_act_idx)
                self.last_act_idx += 1

                data_sample = track.getDataSample().reshape(1, -1)
                print data_sample.shape
                pred = self.labels[self.clf.predict(data_sample)[0]]

                cv2.putText(self.frame,str(pred),(p[0],p[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

                print "Prediction:", pred
                # cv2.imwrite("test.png",self.frame)
                track.delFirst()


        return self.frame

    def drawTracks(self):
        cv2.polylines(self.frame, [np.int32(tr.getTrack()) for tr in self.tracks], False, (0, 255, 0))
        self.frame = cv2.resize(self.frame, (400, 400))
        self.frame = self.predActions()
        cv2.imshow("Video", self.frame)
        return self.frame



 
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
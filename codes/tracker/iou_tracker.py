"""
This is the core tracking engine of IOU+Kalman filter based tracking of bounding box.

An IOU based tracker explained at following link was modified to implement Kalman filter based
data association and tracking.
## https://github.com/bochinski/iou-tracker/blob/master/iou_tracker.py
## https://www.youtube.com/watch?v=JdzaKqXYlnM

"""
import itertools
import numpy as np
import sys

sys.path.append('..')
from kalmanFilter import kf
# from motionModel import constantVelocity
from motionModel import constantAcceleration


class VehicleTracker:

    def __init__(self):
        self.Ta = []
        self.id = itertools.count(1)

    def track_iou(self, detections, time_stamp, sigma_iou, t_min):
        """
        Simple IOU based tracker.
        Ref. "High-Speed Tracking-by-Detection Without Using Image Information by E. Bochinski, V. Eiselein, T. Sikora" for
        more information. Kalman filter has been added by Harsh Nandan.
        Args:
            detections (np.ndarray): array of detections per frame
            time_stamp (float): time stamp of current frame
            sigma_iou (float): IOU threshold.
            t_min (float): minimum track length in frames (currently unused).
        """
        print(' ')
        print('Time stamp: ', time_stamp)

        for track_idx, track in enumerate(self.Ta):
            #! always predict to advance x & P, even if no detection matches
            track['kf'].predict(time_stamp)

            if len(detections) > 0:

                #! get both predicted mean & covariance for gating
                x_pred, P_pred = track['kf'].predict_data_association(time_stamp)

                #! extract center & size from predicted mean
                cg = x_pred[0:2]
                w  = x_pred[2]
                h  = x_pred[3]

                # build predicted bbox from center+size
                predicted_bbox = np.array([
                    cg[0] - w/2, cg[1] - h/2,
                    cg[0] + w/2, cg[1] + h/2
                ], dtype=np.int32)
                track['predicted_box'].append(predicted_bbox)

                #! precompute innovation covariance for Mahalanobis gating
                H     = track['kf'].motion_model.H
                R     = track['kf'].motion_model.R
                S     = H.dot(P_pred).dot(H.T) + R    #! use P_pred, not old P
                S_inv = np.linalg.inv(S)             #! inverse for gating

                # compute IOU (or optionally Mahalanobis) for each detection
                iou_arr_kf_predicted = np.zeros((detections.shape[0], 1))
                for i in range(detections.shape[0]):
                    det = detections[i]
                    # if you want Mahalanobis‐gating, you could do:
                    # z = np.vstack(( self.box_cg(det),
                    #                 [[det[2]-det[0]], [det[3]-det[1]] ]))
                    # nu = z - H.dot(x_pred)
                    # d2 = float(nu.T.dot(S_inv).dot(nu))
                    # if d2 > threshold: continue   #! gated out
                    iou_arr_kf_predicted[i] = self.iou(predicted_bbox, det)

                best_filtered_index = np.argmax(iou_arr_kf_predicted)
                best_filtered_iou   = iou_arr_kf_predicted[best_filtered_index]

                # choose best match purely on IOU (or combine with Mahalanobis)
                if best_filtered_iou > sigma_iou:
                    best_match = detections[best_filtered_index]
                    cg = self.box_cg(best_match)

                    #! only update here—no second predict
                    # track['kf'].predict(time_stamp)  #! removed redundant predict
                    observation = np.vstack((
                        self.box_cg(best_match),
                        np.array([[best_match[2] - best_match[0]],
                                [best_match[3] - best_match[1]]])
                    ))
                    track['kf'].update(observation)

                    track['bboxes'].append(best_match)
                    track['cg'].append(cg)
                    self.Ta[track_idx] = track

                    # remove the matched detection
                    detections = np.delete(detections, best_filtered_index, axis=0)

        # create new tracks from leftover detections (unchanged)
        if len(detections) > 0:
            print('------ starting new track ------ ')
            print('new detections', detections)
            new_tracks = []
            for det in detections:
                print(self.box_cg(det).shape)
                observation = np.vstack((
                    self.box_cg(det),
                    np.array([[det[2]-det[0]], [det[3]-det[1]]])
                ))
                new_tracks.append({
                    'bboxes': [det],
                    'predicted_box': [np.array(det)],
                    'cg': [self.box_cg(det)],
                    'kf': kf.KalmanFilter(
                        observation, time_stamp,
                        constantAcceleration.ConstantAccelerationModel(dims=4)
                    ),
                    'id': next(self.id)
                })
            self.Ta += new_tracks


    def iou(self, bbox1, bbox2):
        """
        Calculates the intersection-over-union of two bounding boxes.
        Args:
            bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
            bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        Returns:
            int: intersection-over-onion of bbox1, bbox2
        """

        bbox1 = [float(x) for x in bbox1]
        bbox2 = [float(x) for x in bbox2]

        (x0_1, y0_1, x1_1, y1_1) = bbox1
        (x0_2, y0_2, x1_2, y1_2) = bbox2

        # get the overlap rectangle
        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)

        # check if there is an overlap
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0

        # if yes, calculate the ratio of the overlap to each ROI size and the unified size
        size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
        size_union = size_1 + size_2 - size_intersection

        return size_intersection / size_union

    def box_cg(self, box):
        return np.array([[(box[0] + box[2]) / 2], [(box[1] + box[3]) / 2]])

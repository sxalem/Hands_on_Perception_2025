import numpy as np

''' A generic Kalman filter which uses an arbitrary linear model specified by the user'''
class KalmanFilter:

    def __init__(self, init_state, time_stamp, motion_model):
        """
        Initialize the kalman filter
        :param init_state: the initial measurement is set as the initial state
        :param time_stamp: the time stamp of the first measurement
        :param motion_model: motion model object
        """
        dt = time_stamp / 1000
        #! ensure we track timestamps for predict_data_association
        self.time_stamp = [time_stamp]
        self.motion_model = motion_model

        self.motion_model.update_F(dt)
        self.motion_model.set_init_state(init_state)

    def predict(self, time_stamp):
        """
        Performs the prediction step of kalman filter
        :param time_stamp: Time stamp at which prediction is asked for
        :return: None
        """
        self.time_stamp += [time_stamp]
        dt = (self.time_stamp[-1] - self.time_stamp[-2]) / 1000
        self.motion_model.update_F(dt)

        # propagate mean normally
        self.motion_model.x = np.matmul(self.motion_model.F, self.motion_model.x) + self.motion_model.u
        #! modified to include process‐noise Q so covariance grows each predict
        self.motion_model.P = (
            self.motion_model.F.dot(self.motion_model.P).dot(self.motion_model.F.T)
            + self.motion_model.Q
        )

    def predict_data_association(self, time_stamp):
        """
        Predict state & covariance for gating, without altering filter state
        :param time_stamp: Time stamp at which prediction is asked for
        :return: (x_pred, P_pred)
        """
        dt = (time_stamp - self.time_stamp[-1]) / 1000
        self.motion_model.update_F(dt)

        # predicted mean
        x_pred = np.matmul(self.motion_model.F, self.motion_model.x) + self.motion_model.u
        #! modified to include Q so gating uses correct, inflated covariance
        P_pred = (
            self.motion_model.F.dot(self.motion_model.P).dot(self.motion_model.F.T)
            + self.motion_model.Q
        )
        #! now returns both predicted mean and covariance
        return x_pred, P_pred

    def update(self, z_measured):
        """
        Performs measurement update
        :param z_measured: position measurement
        :return: None
        """
        z = z_measured
        H = self.motion_model.H
        P = self.motion_model.P
        R = self.motion_model.R

        err = z - H.dot(self.motion_model.x)
        S = H.dot(P).dot(H.T) + R
        S_inv = np.linalg.inv(S)
        K = P.dot(H.T).dot(S_inv)

        self.motion_model.x = self.motion_model.x + K.dot(err)
        self.motion_model.P = (np.eye(self.motion_model.x.shape[0]) - K.dot(H)).dot(P)

import cv2
import numpy as np

class Stabiliser:
    def __init__(self, total_frames, size):
        self.size = size
        self.previous_frame = None
        self.transforms = np.zeros((total_frames, 3), np.float32)

    def movingAverage(self, curve, radius):
        window_size = 2 * radius + 1
        # Define the filter
        f = np.ones(window_size)/window_size
        # Add padding to the boundaries
        curve_pad = np.lib.pad(curve, (radius, radius), 'edge') 
        # Apply convolution
        curve_smoothed = np.convolve(curve_pad, f, mode='same') 
        # Remove padding
        curve_smoothed = curve_smoothed[radius:-radius]
        # return smoothed curve
        return curve_smoothed

    def smooth(self, trajectory):
        smoothed_trajectory = np.copy(trajectory)
        # Filter the x, y and angle curves
        for i in range(3):
                smoothed_trajectory[:,i] = self.movingAverage(trajectory[:,i], radius=50)

        return smoothed_trajectory

    def fixBorder(self, frame):
        s = frame.shape
        # Scale the image 4% without moving the center
        T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
        frame = cv2.warpAffine(frame, T, (s[1], s[0]))
        return frame

    def get_transform(self, frame, i):
        # get first_frame
        if self.previous_frame is None:
            self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            # detect feature points
            prev_pts = cv2.goodFeaturesToTrack(self.previous_frame,
                                               maxCorners=200,
                                               qualityLevel=0.01,
                                               minDistance=30,
                                               blockSize=3)

            # convert current frame to gray
            current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.previous_frame, current, prev_pts, None)

            # check
            assert prev_pts.shape == curr_pts.shape

            # filter points matching points
            idx = np.where(status==1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            # # find transfrom matrix
            transform = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]

            # extract translation
            dx = transform[0,2]
            dy = transform[1,2]

            # extract rotation angle
            da = np.arctan2(transform[1,0], transform[0,0])

            # add to transform list
            self.transforms[i] = [dx, dy, da]

            self.previous_frame = current

    def get_trajectory(self):
        # Find the cumulative sum of tranform matrix for each dx,dy and da
        trajectory = np.cumsum(self.transforms, axis=0)

        # smooth
        smoothed_trajectory = self.smooth(trajectory)
        difference = smoothed_trajectory - trajectory
        self.transforms_smooth = self.transforms + difference
        return trajectory, smoothed_trajectory

    def get_stable_frame(self, frame, i):
        # Extract transformations from the new transformation array
        dx = self.transforms_smooth[i,0]
        dy = self.transforms_smooth[i,1]
        da = self.transforms_smooth[i,2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2,3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, self.size)

        # Fix border artifacts
        frame_stabilized = self.fixBorder(frame_stabilized)

        return frame_stabilized

import cv2
import numpy as np
import math


class FrameEditor:
    """A class for editing framesa

        Takes gamma value as parameter

        Todo add variable function for LUT
    """
    def __init__(self, gamma=1.5):
        self.gamma = gamma
        # get lookup table
        self.lookup = np.array([math.log(i, 10) * 106 for i in np.arange(1, 257)], dtype=np.uint8)

    def normalise(self, frame):
        # check for overflow returns float type to int type
        if frame.max() > 255:
            frame = frame * (255 / gamma_corrected_f.max())
        frame_i = frame.astype("uint8")
        return frame_i

    def doOperation(self, frame):
        # gamma correction
        # frame is normalised and returned to int format
        gamma_corrected = self.normalise(np.power(frame/ 255, self.gamma) * 255)
        # apply lookup table
        # only accepts uint
        edited = cv2.LUT(gamma_corrected, self.lookup)
        # return gamma again normalised
        gamma_final = self.normalise(np.power(edited / 255, 1 / self.gamma) * 255)
        return gamma_final


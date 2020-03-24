import cv2
import numpy as np


class MovingAverage:
    def __init__(self, size, include_motion, buffer_size=30):
        self.include_motion = include_motion
        self.size = size
        self.edit_queue = []
        self.motion_queue = []
        self.buffer_size = buffer_size

    def add(self, frame, motion):
        averageFrame = None

        # check buffer isnt full
        if len(self.edit_queue) >= self.buffer_size:

            # get blended frame
            averageFrame = self.average()

        # append new frames
        self.motion_queue.append(motion)
        self.edit_queue.append(frame)

        return averageFrame

    def clear(self):
        self.edit_queue = []
        self.motion_queue = []

    def get_buffer_size(self):
        return self.buffer_size

    def average(self):
        # uint16 arrays to avoid overflow
        average = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint16)
        maskTotal = np.zeros((self.size[1], self.size[0]), dtype=np.uint16)

        # loop through each frame and its corresponding mask in the queue
        for frame, mask in zip(self.edit_queue, self.motion_queue):
            # get stationary contribution from frame
            contribution = cv2.bitwise_and(frame, frame, mask=mask)

            # combine each contribution in buffer
            average = cv2.add(average, contribution, dtype=18)

            # combine mask values for total contribution per pixel
            maskTotal = cv2.add(maskTotal, mask, dtype=2)

        # scale mask to count of motion frames
        # i.e. all buffer frames do not include motion
        maskTotal = (maskTotal / 255).astype("uint8")

        # make 3 channel
        maskTotal = cv2.merge((maskTotal, maskTotal, maskTotal))

        # remove final first frame from queue
        outFrame = self.edit_queue.pop(0)

        # use motion as current mask
        outMask = self.motion_queue.pop(0)

        # divide per each contributing frame
        final = cv2.divide(average, maskTotal, dtype=16)

        if self.include_motion:
            # initial contribution
            masked_motion = cv2.bitwise_and(outFrame, outFrame, mask=~outMask)

            # mask areas of motion
            masked_static = cv2.bitwise_and(final, final, mask=outMask)

            # blend together
            final = cv2.addWeighted(masked_motion, 0.5, masked_static, 1.0, 0)

        # return frame divided by
        return final

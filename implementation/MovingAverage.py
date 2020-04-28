import cv2
import numpy as np


class MovingAverage:
    def __init__(self, size, include_motion, luminance, buffer_size=30):
        self.include_motion = include_motion
        self.luminance = luminance
        self.size = size
        self.edit_queue = []
        self.motion_queue = []
        self.buffer_size = buffer_size

    def add(self, frame, motion):
        average_frame = None

        # make luminance change
        if self.luminance:
            frame = self.get_chroma(frame)

        # check buffer isnt full
        if len(self.edit_queue) >= self.buffer_size:

            # get blended frame
            average_frame = self.average()

            # remove luminance change
            if self.luminance:
                average_frame = self.invert_chroma(average_frame)

        # append new frames
        self.motion_queue.append(motion)
        self.edit_queue.append(frame)

        return average_frame

    def get_chroma(self, frame):
        # split frame into b g r
        chan = cv2.split(frame)
        # sum each channel b + g + r
        summed = cv2.add(cv2.add(chan[2], chan[1], dtype=cv2.CV_32F), chan[0], dtype=cv2.CV_32F)
        # create 2 channel rep so divide possible
        full = cv2.merge((summed, summed))
        # get chroma - only need two channels get third with 1 - r - g
        chroma = cv2.divide(cv2.merge((chan[0], chan[1])), full, dtype=cv2.CV_32F)
        # get lightness
        lightness = cv2.divide(summed, 3, dtype=cv2.CV_32F)
        # compile into 3 channel
        merged = cv2.merge((chroma, lightness))
        # return merged
        return merged

    def invert_chroma(self, frame):
        # split frame into b g r
        chan = cv2.split(frame)
        # average lightness x 3
        lightness = chan[2] * 3
        # 3 channel rep
        lightness_3c = cv2.merge((lightness, lightness, lightness))
        # red
        red = 1 - chan[0] - chan[1]
        # average chroma
        chroma = cv2.merge((chan[0], chan[1], red))
        # return to value
        average_frame = cv2.multiply(chroma, lightness_3c)
        return average_frame

    def clear(self):
        self.edit_queue = []
        self.motion_queue = []

    def get_buffer_size(self):
        return self.buffer_size

    def average(self):
        # uint16 arrays to avoid overflow
        average = np.zeros((*self.size, 3), dtype=np.float32)
        maskTotal = np.zeros((self.size), dtype=np.uint16)

        # loop through each frame and its corresponding mask in the queue
        for frame, mask in zip(self.edit_queue, self.motion_queue):

            # combine each contribution in buffer
            average = cv2.add(average, frame, dtype=21)

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
        final = cv2.divide(average, maskTotal, dtype=21)

        if self.include_motion:
            # initial contribution
            masked_motion = cv2.bitwise_and(outFrame, outFrame, mask=~outMask)

            # mask areas of motion
            masked_static = cv2.bitwise_and(final, final, mask=outMask)

            # blend together
            final = cv2.addWeighted(masked_motion, 0.5, masked_static, 1.0, 0)

        # return frame divided by
        return final

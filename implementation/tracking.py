import cv2
import numpy as np
import math

class MotionTracker:
    def __init__(self):
        self.m = None
        self.buffer = 50
        self.queue = []
        self.sigMatrixRed = None
        self.sigMatrixBlue = None

    def getGaussian(self, x, sigma):
        eMatrix = np.ones((x.shape)) * e
        left = 1 / (sigma * math.sqrt(2 * pi))
        power = ((-1) * cv2.pow(x, 2)) / (2 * cv2.pow(sigma, 2))
        right = np.power(eMatrix, power)
        return left * right

    def getSize(self):
        return len(self.queue)

    def updateSig(self):
        previous = self.queue[0]
        differences = []
        for frame in self.queue[1:]:
            diff = cv2.absdiff(frame, previous)
            testchrom = self.get_chroma(diff)
            differences.append(diff)
            previous = frame
        differences = np.array(differences)
        medians = np.median(differences, axis=0)
        r_med, b_med = cv2.split(medians)
        self.sigMatrixRed = r_med / ( 0.68 * math.sqrt(2))
        self.sigMatrixBlue = b_med / ( 0.68 * math.sqrt(2))

    def get_chroma(self, chroma):
        chan = cv2.split(chroma)
        cha3 = np.ones((chan[0].shape), dtype=np.float32) - chan[0] - chan[1]
        return cv2.merge((chan[0], chan[1], cha3))

    def add(self, chroma, lightness):
        # if enough to get motion
        if len(self.queue) > self.buffer:
            self.updateSig()
            # check background
            total = np.zeros((720, 1280), dtype=np.float32)
            first = True
            for f in self.queue:
                # get r and g chroma values
                current_split = cv2.split(chroma)
                # get background values
                test = self.get_chroma(f)
                background_split = cv2.split(f)
                # get frame differences
                r_diff = current_split[0] - background_split[0]
                b_diff = current_split[1] - background_split[1]
                # red gauss
                r_gaussian = self.getGaussian(r_diff, self.sigMatrixRed)
                b_gaussian = self.getGaussian(b_diff, self.sigMatrixBlue)
                # accumulate frames
                total += (r_gaussian * b_gaussian)

                # divide by sample size
            prob = total / self.buffer
            prob_bg = (prob * 0.99) / ((prob * 0.99) + (0.001 * (1 - 0.99)))
            _, prob_bg = cv2.threshold(prob_bg, 0.05, 1, cv2.THRESH_BINARY_INV)
            cv2.imshow("p", prob_bg)
            cv2.waitKey(1)
            self.queue.pop(0)
        self.queue.append(chroma)


cap = cv2.VideoCapture("../sources/motion_test.mp4")
pi = math.pi
e = math.e
mt = MotionTracker()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # split frame into b g r
    chan = cv2.split(frame)
    # sum each channel b + g + r
    summed = cv2.add(cv2.add(chan[2], chan[1], dtype=cv2.CV_32F), chan[0], dtype=cv2.CV_32F)
    # add to 3 channel rep so divide possible
    full = cv2.merge((summed, summed))
    # get chroma
    chroma = cv2.divide(cv2.merge((chan[2], chan[1])), full, dtype=cv2.CV_32F)
    # get lightness
    lightness = cv2.divide(summed, 3, dtype=cv2.CV_8U)

    mt.add(chroma, lightness)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

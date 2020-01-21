import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import argparse

# add argument parser for clean use
parser = argparse.ArgumentParser(description="Video Enhancer")
parser.add_argument('-p', '--path', type=str, help='source of video file')
parser.add_argument('-l', '--loop', action='store_true', help='repeat loop of video')
parser.add_argument('-s', '--size', nargs=2,  help='size of video to display')
args = parser.parse_args()


class MovingAverage:
    def __init__(self, first=None):
        self.previous = first
        self.frame = 1

    def add(self, frame):
        current = self.previous + ((frame - self.previous) / (self.frame + 1))
        self.previous = current
        self.frame += 1
        print(self.frame)

    def get_current(self):
        return self.previous


def getVideo(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("No Video Found")
        return None
    else:
        return cap

def loopVideo(cap, size, loop):
    count = 0
    _, first = cap.read()
    ma = MovingAverage(first)
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # store frame in moving average
            ma.add(frame)
            smoothFrame = ma.get_current() / 255.0
            # resize
            frame = cv2.resize(frame, (size[0], size[1]))

            # Call image operations here
            editFrame = imageOperations(frame)

            cv2.imshow("MA", smoothFrame)
        else:
            # if looping
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                print("video finished")
                break
        # key to break playback
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def imageOperations(frame):
    # convert to luminance
    luma = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2XYZ))[1]
    return luma

def histogram():
    # get each color plane
    bgr_planes = cv2.split(frame)

    # histogram calculation
    #                     source     , index, mask? , size, range
    b_hist = cv2.calcHist(bgr_planes, [0], None, [256], [0, 256])
    g_hist = cv2.calcHist(bgr_planes, [1], None, [256], [0, 256])
    r_hist = cv2.calcHist(bgr_planes, [2], None, [256], [0, 256])
    l_hist = cv2.calcHist([luma], [0], None, [256], [0, 256])

    # hist setup
    hist_w = int(512 * 0.75)
    hist_h = int(400 * 0.75)
    bin_w = int(round(hist_w / 256))

    histImageRGB = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    histImageLum = np.zeros((hist_h, hist_w, 1), dtype=np.uint8)
    cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h,
                  norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h,
                  norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h,
                  norm_type=cv2.NORM_MINMAX)
    cv2.normalize(l_hist, l_hist, alpha=0, beta=hist_h,
                  norm_type=cv2.NORM_MINMAX)

    for i in range(1, 256):
        cv2.line(histImageRGB,
                 (bin_w * (i - 1),
                  hist_h - int(np.round(b_hist[i - 1]))),
                 (bin_w * (i), hist_h - int(np.round(b_hist[i]))),
                 (255, 0, 0), thickness=2)
        cv2.line(histImageRGB,
                 (bin_w * (i - 1),
                  hist_h - int(np.round(g_hist[i - 1]))),
                 (bin_w * (i), hist_h - int(np.round(g_hist[i]))),
                 (0, 255, 0), thickness=2)
        cv2.line(histImageRGB,
                 (bin_w * (i - 1),
                  hist_h - int(np.round(r_hist[i - 1]))),
                 (bin_w * (i), hist_h - int(np.round(r_hist[i]))),
                 (0, 0, 255), thickness=2)
        cv2.line(histImageLum,
                 (bin_w * (i - 1),
                  hist_h - int(np.round(l_hist[i - 1]))),
                 (bin_w * (i), hist_h - int(np.round(l_hist[i]))),
                 (255, 255, 255), thickness=2)

    # view luma as 3 channels to allow concatenation
    luma_3_channel = cv2.cvtColor(luma, cv2.COLOR_GRAY2BGR)
    combined = np.concatenate((frame, luma_3_channel), axis=1)

    # Display the resulting frame
    cv2.imshow('frame', combined)
    cv2.imshow('histrgb', histImageRGB)
    cv2.imshow('histluma', histImageLum)

def equal():
    # equalise histogram
    eq = cv2.equalizeHist(luma)
    # get hist
    l_hist = cv2.calcHist([eq], [0], None, [256], [0, 256])

    # hist setup
    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w / 256))

    histImageLum = np.zeros((hist_h, hist_w, 1), dtype=np.uint8)

    for i in range(1, 256):
        cv2.line(histImageLum,
                 (bin_w * (i - 1),
                  hist_h - int(np.round(l_hist[i - 1]))),
                 (bin_w * (i), hist_h - int(np.round(l_hist[i]))),
                 (255, 255, 255), thickness=2)

    combined = np.concatenate((luma, eq), axis=1)
    cv2.imshow('frame', combined)
    cv2.imshow('histluma', histImageLum)

def clahe():
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(luma)
    combined = np.concatenate((luma, cl1), axis=1)
    cv2.imshow('frame', combined)


def main(args):
    cap = getVideo(args.path)
    if args.size == None:
        size = [640, 480]
    else:
        size = args.size
    if cap != None:
        loopVideo(cap, size, args.loop)


if __name__ == "__main__":
    # when run as script
    main(args)

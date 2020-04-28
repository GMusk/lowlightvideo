import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def nothing():
    pass

current = "mid-light-train"

videos = ["stable.mp4", "motion.mp4", "closure.mp4", "expand.mp4", "average.mp4"]
caps = []

# Create a black image, a window
cap = cv2.VideoCapture("../sources/" + current + ".mp4")
caps.append(cap)
total_frames = cap.get(7)
print(total_frames)

for video in videos:
    cap = cv2.VideoCapture("../out/" + current + "/" + video)
    caps.append(cap)

cv2.namedWindow('video')

# create trackbars for color change
cv2.createTrackbar('frame', 'video', 0, int(total_frames) - 1,nothing)
cv2.createTrackbar('stream', 'video', 0, len(caps) - 1, nothing)

while 1:
    # get current positions of four trackbars
    f = cv2.getTrackbarPos('frame','video')
    c = cv2.getTrackbarPos('stream','video')

    cap = caps[c]
    cap.set(1, f)

    ret, frame = cap.read()

    if ret:
        h, w, _ = frame.shape

        new_h = int(h / 2)
        new_w = int(w / 2)

        resize = cv2.resize(frame, (new_w, new_h))
    else:
        resize = np.ones((new_h, new_w))

    cv2.imshow('video', resize)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()


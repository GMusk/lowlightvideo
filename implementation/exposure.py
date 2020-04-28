import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

three_frames = [None, None, None]
shutter_speed = np.array([math.log(1/60), math.log(1/125), math.log(1/30)])
count = 0
z = np.empty((0, 3), dtype=np.uint8)
z_min = 0
z_max = 255


def weight(z):
    if z <= 0.5 * (z_min + z_max):
        return z - z_min
    else:
        return z_max - z

def get_response_function(z, shutter_speed, l, w):
    n = 256
    # create a of size pixels and desired output range
    a = np.zeros((z.size + n + 1, z.shape[0] + n))
    b = np.zeros((a.shape[0], 1))

    k = 1
    # for point in selected points
    for i in range(len(z)):
        # for each exposure value
        for j in range(len(shutter_speed)):
            # get the weight of that pixel
            wij = weight(z[i][j] + 1)
            a[k, z[i][j] + 1] = wij
            a[k, n + i] = -wij
            b[k, 0] = wij * shutter_speed[j]
            k += 1

    a[k, 129] = 1
    k=k+1

    for i in range(n - 1):
        a[k, i] = l * weight(i + 1)
        a[k, i + 1] = -2 * l * weight(i + 1)
        a[k, i + 2] = l * weight(i + 1)
        k += 1

    print(a.shape[1])
    print(np.linalg.matrix_rank(a))

    x = np.linalg.lstsq(a, b, rcond=None)[0]

    g = x[:n]
    l_e = x[n:]

    return g, l_e

def nothing(x):
    pass

def select_exposure_frame(event, x, y, flags, param):
    global count
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("frame" + str(count + 1) + " selected")
        three_frames[count] = frame
        count += 1

def get_pixel_value(event, x, y, flags, param):
    global z
    if event == cv2.EVENT_LBUTTONDBLCLK:
        t = np.zeros((1, 3), dtype=np.uint8)
        t[0][0] = three_frames[0][y][x]
        t[0][1] = three_frames[1][y][x]
        t[0][2] = three_frames[2][y][x]
        z = np.concatenate((z, t), axis=0)
        print(z)

# Create a black image, a window
cap = cv2.VideoCapture("../sources/signal_curve.mp4")
total_frames = cap.get(7)

cap.set(1, 0)
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
split = cv2.split(frame)
three_frames[0] = split[0]
cap.set(1, 275)
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
split = cv2.split(frame)
three_frames[1] = split[0]
cap.set(1, 440)
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
split = cv2.split(frame)
three_frames[2] = split[0]

"""
cv2.namedWindow('video')

# create trackbars for color change
cv2.createTrackbar('frame', 'video', 0, int(total_frames) - 1,nothing)
cv2.setMouseCallback('video', select_exposure_frame)

while count != 3:
    # get current positions of four trackbars
    f = cv2.getTrackbarPos('frame','video')

    cap.set(1, f)

    ret, frame = cap.read()

    cv2.imshow('video',frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()
"""

# Create a black image, a window
cv2.namedWindow('exposures')

# create trackbars for color change
cv2.createTrackbar('frame', 'exposures', 0, 2, nothing)
cv2.setMouseCallback('exposures', get_pixel_value)

while(1):
    # get current positions of four trackbars
    f = cv2.getTrackbarPos('frame','exposures')

    frame = three_frames[f]

    cv2.imshow('exposures',frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()


g, l_e = get_response_function(z, shutter_speed, 1, weight)
y = list(range(256))
plt.plot(g, y)
plt.show()

compare = three_frames[0].copy()
test_frame = three_frames[0].copy()

for i in range(test_frame.shape[0]):
    for j in range(test_frame.shape[1]):
        radiance = 0
        denom = 0
        for e in range(3):
            radiance += weight(three_frames[e][i][j]) * (g[three_frames[e][i][j]] - shutter_speed[e])
            denom += weight(three_frames[e][i][j])
        r = math.log(radiance / denom)
        if math.isnan(r):
            test_frame[i][j] = 255
        else:
            test_frame[i][j] = 128 * r

cv2.imshow("compare", compare)
cv2.imshow("test", test_frame)
cv2.waitKey(0)

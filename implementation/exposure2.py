import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

three_frames = [None, None, None]
shutter_speed = np.array([math.log(1/60), math.log(1/125), math.log(1/30)])
count = 0
zb = np.empty((0, 3), dtype=np.uint8)
zg = np.empty((0, 3), dtype=np.uint8)
zr = np.empty((0, 3), dtype=np.uint8)
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
    global zb, zg, zr
    if event == cv2.EVENT_LBUTTONDBLCLK:
        t = np.zeros((1, 3), dtype=np.uint8)
        t[0][0] = three_frames[0][y][x][0]
        t[0][1] = three_frames[1][y][x][0]
        t[0][2] = three_frames[2][y][x][0]
        zb = np.concatenate((zb, t), axis=0)
        t = np.zeros((1, 3), dtype=np.uint8)
        t[0][0] = three_frames[0][y][x][1]
        t[0][1] = three_frames[1][y][x][1]
        t[0][2] = three_frames[2][y][x][1]
        zg = np.concatenate((zg, t), axis=0)
        t = np.zeros((1, 3), dtype=np.uint8)
        t[0][0] = three_frames[0][y][x][2]
        t[0][1] = three_frames[1][y][x][2]
        t[0][2] = three_frames[2][y][x][2]
        zr = np.concatenate((zr, t), axis=0)

# Create a black image, a window
cap = cv2.VideoCapture("../sources/signal_curve.mp4")
total_frames = cap.get(7)

cap.set(1, 0)
ret, frame = cap.read()
three_frames[0] = frame
cap.set(1, 275)
ret, frame = cap.read()
three_frames[1] = frame
cap.set(1, 440)
ret, frame = cap.read()
three_frames[2] = frame

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


gb, l_e = get_response_function(zb, shutter_speed, 1, weight)
gg, l_e = get_response_function(zg, shutter_speed, 1, weight)
gr, l_e = get_response_function(zr, shutter_speed, 1, weight)
y = list(range(256))
plt.plot(gb, y)
plt.show()

compare = three_frames[0].copy()
test_frame = three_frames[0].copy()

for i in range(test_frame.shape[0]):
    for j in range(test_frame.shape[1]):
        radiance_b = 0
        radiance_g = 0
        radiance_r = 0
        denom_b = 0
        denom_g = 0
        denom_r = 0
        for e in range(3):
            radiance_b += weight(three_frames[e][i][j][0]) * (gb[three_frames[e][i][j][0]] - shutter_speed[e])
            radiance_g += weight(three_frames[e][i][j][1]) * (gg[three_frames[e][i][j][1]] - shutter_speed[e])
            radiance_r += weight(three_frames[e][i][j][2]) * (gr[three_frames[e][i][j][2]] - shutter_speed[e])
            denom_b += weight(three_frames[e][i][j][0])
            denom_g += weight(three_frames[e][i][j][1])
            denom_r += weight(three_frames[e][i][j][2])
        rb = math.log(radiance_b / denom_b)
        rg = math.log(radiance_g / denom_g)
        rr = math.log(radiance_r / denom_r)
        pixel = np.empty((3))
        if math.isnan(rb):
            pixel[0] = 255
        else:
            pixel[0] = 100 * rb
        if math.isnan(rg):
            pixel[1] = 255
        else:
            pixel[1] = 100 * rg
        if math.isnan(rr):
            pixel[2] = 255
        else:
            pixel[2] = 100 * rr
        test_frame[i][j] = pixel

cv2.imshow("compare", compare)
cv2.imwrite("input.png", compare)
cv2.imshow("test", test_frame)
cv2.imwrite("final.png", test_frame)
cv2.waitKey(0)

import numpy as np
import cv2
from matplotlib import pyplot as plt
# import tools

PATH = "20191228_162636.mp4"
LOOPING = True

# if not path then use webcam
if PATH == "":
    PATH = 0

# 0 for webcam or path to file
cap = cv2.VideoCapture(PATH)
cap.set(3, 640)
cap.set(4, 480)

count = 0

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:

        frame = cv2.resize(frame, (640, 480))
        # Our operations on the frame come here

        # convert to luminance
        luma = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2XYZ))[1]

        if (count % 3) == 0:
            # ===================== histogram stuff =============================
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

            # ===============================================================================

            # 3 channel for concatenation
            luma_3_channel = cv2.cvtColor(luma, cv2.COLOR_GRAY2BGR)
            combined = np.concatenate((frame, luma_3_channel), axis=1)

            # luminance = tools.get_luminance_map(frame)
            # hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

            # plt.hist(gray.ravel(), 256, [0, 256])
            # plt.show()

            # plt.subplot(221), plt.imshow(gray, 'gray')
            # plt.subplot(222), plt.imshow(hist, 'gray')

            # Display the resulting frame
            cv2.imshow('frame', combined)
            cv2.imshow('histrgb', histImageRGB)
            cv2.imshow('histluma', histImageLum)
        elif (count % 3) == 1:
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
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl1 = clahe.apply(luma)
            combined = np.concatenate((luma, cl1), axis=1)
            cv2.imshow('frame', combined)
    else:
        # if looping
        if LOOPING:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print(count)
            count += 1
        else:
            print("video finished")
            break
    # key to break playback
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# implementation of median cut

import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize)


def pix_to_intens(pix):
    return (0.2125 * pix[0]) + (0.7154 * pix[1]) + (0.0721 * pix[2])


def get_luminance_map(img):
    shape = img.shape
    luminance = np.zeros((shape[0], shape[1]))

    # iterate through image converting to intensity
    for i, row in enumerate(img):
        luminance[i] = np.apply_along_axis(pix_to_intens, 1, row)

    return luminance


def get_summed_area(img):
    # get shape for dynamic allocation
    shape = img.shape
    summed_table = np.zeros((shape[0], shape[1]))

    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            current_sum = pixel
            if i != 0:
                current_sum += summed_table[i - 1][j]
            if j != 0:
                current_sum += summed_table[i][j - 1]
            if i != 0 and j != 0:
                current_sum -= summed_table[i - 1][j - 1]
            summed_table[i][j] = current_sum
    return summed_table

def main():

    # read image
    img = cv2.imread("lena(50,50).png")

    # display image
    # cv2.imshow('image', img)

    # wait (x) milliseconds or keypress, 0 wait for keypress
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()

    # cv stores colour data as BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # form image with 1 channel
    luminance = get_luminance_map(img)

    summed_table = get_summed_area(luminance)

    k = 2

    v = img.shape[0]
    h = img.shape[1]
    regions = [((0, 0), (h - 1, v - 1))]

    while k != 0:
        # iterate through region list
        new_regions = []
        for region in regions:
            # get dimensions and total image intensity
            h1 = region[0][0]
            v1 = region[0][1]
            h2 = region[1][0]
            v2 = region[1][1]
            total_intensity = (summed_table[v2][h2]) // 2
            least_diff = total_intensity
            # get longest dimension
            if (h2 - h1) >= (v2 - v1):
                # iterate horizontally
                i = h1
                past_half = False
                while not past_half:
                    value = summed_table[v2][i]
                    i += 1
                    difference = total_intensity - value
                    if difference < 0:
                        past_half = True
                        new_regions.append(((h1, v1), (i, v2)))
                        new_regions.append(((i, v1), (h2, v2)))
                    else:
                        if difference < least_diff:
                            least_diff = difference
            else:
                i = v1
                past_half = False
                while not past_half:
                    value = summed_table[i][h2]
                    i += 1
                    difference = total_intensity - value
                    if difference < 0:
                        past_half = True
                        new_regions.append(((h1, v1), (h2, i)))
                        new_regions.append(((h1, i), (h2, v2)))
                    else:
                        if difference < least_diff:
                            least_diff = difference
        regions = new_regions
        print(regions)
        k -= 1

    for i in range(len(regions) // 2):
        img = cv2.line(img, regions[i][1], regions[i + 1][0], (255, 0, 0), 1)

    plt.imshow(img)
    plt.show()

if __name__== "__main__":
  main()
from __future__ import division
import cv2
import numpy as np
import argparse
from skimage.draw import line, polygon
import glob, os
import json
import imutils
import matplotlib.pyplot as plt


def getPredictions(image_file, tube_coords_json, plotting):
    image = cv2.imread(image_file)  # image is loaded as BGR
    tube_coords = json.loads(tube_coords_json)
    f = open(image_file + ".coords.txt", "w")
    f.write(tube_coords_json)
    f.close()
    strip_count = len(tube_coords) - 1
    # Filter the image to enhance various features
    # image = applyClahetoRGB(image, cv2.COLOR_BAYER_BG2RGB)  # Increase contrast to the image

    unstandardized_scores = [None] * strip_count
    tmp = image.copy()
    tmp_filtered = image.copy()

    for i in range(0, strip_count):

        tube_width = int(((tube_coords[i][2] - tube_coords[i][0]) ** 2 + (
                tube_coords[i][3] - tube_coords[i][1]) ** 2) ** (1 / 2))
        # let's get background intensity so we can normalize the signal from the fluorescent liquid
        box = np.zeros((5, 2))
        box[0] = np.asarray([tube_coords[i][0], tube_coords[i][1]])  # top right
        box[1] = np.asarray([tube_coords[i + 1][0], tube_coords[i + 1][1]])  # bottom right
        box[2] = np.asarray(
            [tube_coords[i][0] - tube_width / 2, tube_coords[i + 1][1]])  # bottom left
        box[3] = np.asarray(
            [tube_coords[i + 1][0] - tube_width / 2, tube_coords[i][1]])  # top left
        box[4] = np.asarray([tube_coords[i][0], tube_coords[i][1]])  # top right
        if plotting:
            tmp = cv2.drawContours(tmp, [np.array(box[0:4]).reshape((-1, 1, 2)).astype(np.int32)],
                                   0, (255, 0, 0), 2)
        rr, cc = polygon(box[:, 0], box[:, 1])
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask[cc, rr] = 1
        subimage2 = cv2.bitwise_and(image, image, mask=mask)
        # subimage2[:, :, 0] = np.zeros([subimage2.shape[0], subimage2.shape[1]])
        bkgd_red = np.median(subimage2[cc, rr, 2])  # (np.sum(subimage2[:, :, 2])) / np.sum(mask)
        bkgd_grn = np.median(subimage2[cc, rr, 1])  # (np.sum(subimage2[:, :, 1])) / np.sum(mask)
        bkgd_blu = np.median(subimage2[cc, rr, 0])  # (np.sum(subimage2[:, :, 0])) / np.sum(mask)

        # In theory, tlx and brx values don't need to be arrays. However, when we add support for
        # rotated boxes, we will need array support anyways.
        # for plottindasdg purposes, define the 4 corners of this tube's enclosing area.
        box = np.zeros((5, 2))
        box[0] = np.asarray([tube_coords[i][0], tube_coords[i][1]])  # top left
        box[1] = np.asarray([tube_coords[i][2], tube_coords[i][3]])  # top right
        box[2] = np.asarray([tube_coords[i + 1][2], tube_coords[i + 1][3]])  # bottom right
        box[3] = np.asarray([tube_coords[i + 1][0], tube_coords[i + 1][1]])  # bottom left
        box[4] = np.asarray([tube_coords[i][0], tube_coords[i][1]])  # top left
        rr, cc = polygon(box[:, 0], box[:, 1])
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask[cc, rr] = 1
        # focus in on the tube liquid's enclosing area
        subimage = cv2.bitwise_and(image, image, mask=mask)
        blue_cutoff = 50
        b, g, r = cv2.split(subimage)
        blue_mask = b[:, :] > blue_cutoff
        g[blue_mask] = 0
        r[blue_mask] = 0
        # blue channel is all noise, so get rid of it:
        b[:, :] = np.zeros([b.shape[0], b.shape[1]])
        # subtract away background noise level
        g_mask = g[:, :] < bkgd_grn.astype("uint8") + 1
        r_mask = r[:, :] < bkgd_red.astype("uint8") + 1
        g -= bkgd_grn.astype("uint8")
        r -= bkgd_red.astype("uint8")
        g[g_mask] = 0
        r[r_mask] = 0
        g[blue_mask] = np.mean(g[cc, rr])
        r[blue_mask] = np.mean(r[cc, rr])
        subimage = cv2.merge([b, g, r])
        # We want to get signal from the part of the tube which contains liquid, and not any other
        # background signal. As such, we model the bottom of the tube as a trapezoid and create a
        # kernel to traverse through the tube's enclosing area to find the portion with the highest
        # signal.
        tube_height = int(((box[3][0] - box[0][0]) ** 2 + (box[3][1] - box[0][1]) ** 2) ** (1 / 2))
        angle = np.rad2deg(np.arctan2(box[0][1] - box[1][1], box[1][0] - box[0][0]))
        kernel = create_kernel(tube_width, tube_height, angle, plotting)
        blurs_green = cv2.filter2D(subimage[:, :, 1], -1, kernel)
        blurs_red = cv2.filter2D(subimage[:, :, 2], -1, kernel)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(blurs_green + blurs_red)
        if plotting:
            tmp = cv2.drawContours(tmp, [np.array(box[0:4]).reshape((-1, 1, 2)).astype(np.int32)],
                                   0, (0, 0, 255), 2)
            plt.hist(subimage.ravel(), 256, [0, 256], log=True)
            plt.title('tube {}\n{}'.format(i, image_file.split('\\')[-1]))
            plt.show()

        unstandardized_scores[i] = abs(maxVal)
        # unstandardized_scores[i] = maxVal

    if plotting:
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
        plt.title('{}'.format(image_file.split('\\')[-1]))
        plt.show()

    final_score = [unstandardized_score / unstandardized_scores[-1]
                   for unstandardized_score in unstandardized_scores]

    f = open(image_file + ".scores.txt", "w")
    f.write(json.dumps(final_score))
    f.close()

    return final_score


def create_kernel(tube_width, tube_height, angle, plotting):
    """
    :return: a trapezoidal kernel
    kernel_height = int(tube_height
    """
    kernel_width = tube_width
    kernel_height = tube_height
    kernel = np.zeros((kernel_height, kernel_width), np.float32)
    trap_height_large = int(38 * tube_height / 75)
    trap_height_small = int(18 * tube_height / 75)
    trap_length = int(55 * tube_width / 70)
    trapezoid = np.zeros((4, 2))
    trapezoid[0] = np.asarray([kernel_height / 2 - trap_height_small / 2, 0])  # top left
    trapezoid[1] = np.asarray([kernel_height / 2 - trap_height_large / 2, trap_length])  # top right
    trapezoid[2] = np.asarray(
        [kernel_height / 2 + trap_height_large / 2, trap_length])  # bottom right
    trapezoid[3] = np.asarray([kernel_height / 2 + trap_height_small / 2, 0])  # bottom left
    rr, cc = polygon(trapezoid[:, 0], trapezoid[:, 1], kernel.shape)
    kernel[rr, cc] = 1
    kernel = kernel / cv2.sumElems(kernel)[0]
    kernel = imutils.rotate_bound(kernel, -1 * angle)
    # if plotting:
    #     plt.figure()
    #     plt.imshow(kernel)
    #     plt.show()

    return kernel


# from https://stackoverflow.com/a/37123933
def check(p1, p2, base_array):
    """
    Uses the line defined by p1 and p2 to check array of
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(base_array.shape)  # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    if p1[0] == p2[0]:
        max_col_idx = (idxs[0] - p1[0]) * idxs.shape[1]
        sign = np.sign(p2[1] - p1[1])
    else:
        max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
        sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign


# from https://stackoverflow.com/a/37123933
def create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k - 1], vertices[k], base_array)], axis=0)

    # Set all values inside polygon to one
    base_array[fill] = 1 / (shape[0] * shape[1])

    return base_array


def applyClahetoRGB(bgr_imb):
    lab = cv2.cvtColor(bgr_imb, cv2.COLOR_BGR2LAB)
    # Split lab image to different channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Convert image from LAB Color model to RGB model
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def run_analysis(file, tube_coords, threshold, plotting=False):
    final_scores = getPredictions(file, tube_coords, plotting)
    calls = [1 if x > threshold else 0 for x in final_scores]
    print(json.dumps({"final_scores": final_scores, "calls": calls}))


def main():
    parser = argparse.ArgumentParser('Read strip tubes')
    parser.add_argument('--image_file', required=False)
    parser.add_argument('--tubeCoords', required=False)
    parser.add_argument('--plotting', help="Enable plotting", action='store_true')
    args = parser.parse_args()
    threshold = 2.5

    if True:  # args.image_file is None:
        for file in glob.glob(
                r'C:\Users\Sameed\Documents\Educational\PhD\Rotations\Pardis\SHERLOCK-reader\jon_pictures\uploads\*jpg'):
            print(file)

            tube_coords = None
            with open(file + ".coords.txt") as f:
                for line in f:  # there should only be one line in file f
                    tube_coords = line
            run_analysis(file, tube_coords, threshold, plotting=True)
            print()
    else:
        final_scores = run_analysis(args.image_file, args.tubeCoords, threshold, args.plotting)


if __name__ == '__main__':
    main()

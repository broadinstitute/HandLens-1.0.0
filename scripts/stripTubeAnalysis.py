from __future__ import division
import cv2
import numpy as np
import argparse
from skimage.draw import line, polygon
import matplotlib
import matplotlib.pyplot as plt


def getPredictions(image_file, strip_tl_x, strip_tl_y, strip_br_x, strip_br_y, strip_count,
                   plotting):
    image = cv2.imread(image_file)  # image is loaded as BGR

    # Filter the image to enhance various features
    # image = applyClahetoRGB(image, cv2.COLOR_BAYER_BG2RGB)  # Increase contrast to the image

    unnormalized_scores = [None] * strip_count
    strips_tlx = [None] * strip_count  # top left x-values
    strips_tly = [None] * strip_count  # top right x-values
    strips_brx = [None] * strip_count  # bottom right x-values
    strips_bry = [None] * strip_count  # bottom right x-values

    tube_dx = strip_br_x - strip_tl_x
    tube_bottom_height = (strip_br_y - strip_tl_y) / strip_count

    tmp = image.copy()
    tmp_filtered = image.copy()

    for i in range(0, strip_count):
        # In theory, tlx and brx values don't need to be arrays. However, when we add support for
        # rotated boxes, we will need array support anyways.
        strips_tlx[i] = int(strip_tl_x if i == 0 else strips_tlx[i - 1])
        strips_brx[i] = int(strips_tlx[0] + tube_dx if i == 0 else strips_tlx[i] + tube_dx)
        strips_tly[i] = int(strip_tl_y if i == 0 else strips_bry[i - 1])
        strips_bry[i] = int(
            strips_tly[0] + tube_bottom_height if i == 0 else strips_tly[i] + tube_bottom_height)

        # for plotting purposes, define the 4 corners of this tube's enclosing area.
        box = np.zeros((4, 2))
        box[0] = np.asarray([strips_tlx[i], strips_tly[i]])  # top left
        box[1] = np.asarray([strips_brx[i], strips_tly[i]])  # top right
        box[2] = np.asarray([strips_brx[i], strips_bry[i]])  # bottom right
        box[3] = np.asarray([strips_tlx[i], strips_bry[i]])  # bottom left
        box = np.array(box).reshape((-1, 1, 2)).astype(np.int32)

        # focus in on the tube liquid's enclosing area
        subimage = image[strips_tly[i]:strips_bry[i], strips_tlx[i]:strips_brx[i]]
        # blue channel is all noise, so get rid of it:
        subimage[:, :, 0] = np.zeros([subimage.shape[0], subimage.shape[1]])

        # We want to get signal from the part of the tube which contains liquid, and not any other
        # background signal. As such, we model the bottom of the tube as a trapezoid and create a
        # kernel to traverse through the tube's enclosing area to find the portion with the highest
        # signal.
        kernel = create_kernel(tube_dx, tube_bottom_height, plotting)
        blurs_green = cv2.filter2D(subimage[:, :, 1], -1, kernel)
        blurs_red = cv2.filter2D(subimage[:, :, 2], -1, kernel)

        _, maxVal, _, _ = cv2.minMaxLoc(blurs_green + blurs_red)

        # let's get background intensity so we can normalize the signal from the fluorescent liquid
        bckgrnd_subimage = image[strips_tly[i]:strips_bry[i],
                           int(strips_tlx[i] - tube_dx / 2):strips_tlx[i]]
        unnormalized_scores[i] = maxVal / ((cv2.mean(bckgrnd_subimage[:, :, 1])[0] + cv2.mean(
            bckgrnd_subimage[:, :, 2])[0]) / 2)
        # unnormalized_scores[i] = maxVal
        if plotting:
            tmp[strips_tly[i]:strips_bry[i], strips_tlx[i]:strips_brx[i], 1] = blurs_green
            tmp[strips_tly[i]:strips_bry[i], strips_tlx[i]:strips_brx[i], 2] = blurs_red
            tmp = cv2.drawContours(tmp, [box], 0, (0, 0, 255), 2)

    if plotting:
        fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
    plt.show()

    thresh = 1.2
    final_score = [score / unnormalized_scores[-1] for score in unnormalized_scores]
    print(final_score)
    return ["Positive" if score > thresh else "Negative" for score in final_score[0:-1]]


def create_kernel(tube_dx, tube_bottom_height, plotting):
    """
    :return: a trapezoidal kernel
    """
    kernel_height = int(tube_bottom_height)
    kernel_width = int(tube_dx)
    kernel = np.zeros((kernel_height, kernel_width), np.float32)
    trap_height_large = int(38 * tube_bottom_height / 75)
    trap_height_small = int(18 * tube_bottom_height / 75)
    trap_length = int(55 * tube_dx / 70)
    trapezoid = np.zeros((4, 2))
    trapezoid[0] = np.asarray([kernel_height / 2 - trap_height_small / 2, 0])  # top left
    trapezoid[1] = np.asarray([kernel_height / 2 - trap_height_large / 2, trap_length])  # top right
    trapezoid[2] = np.asarray(
        [kernel_height / 2 + trap_height_large / 2, trap_length])  # bottom right
    trapezoid[3] = np.asarray([kernel_height / 2 + trap_height_small / 2, 0])  # bottom left
    rr, cc = polygon(trapezoid[:, 0], trapezoid[:, 1], kernel.shape)
    kernel[rr, cc] = 1
    kernel = kernel / cv2.sumElems(kernel)[0]
    if plotting:
        plt.imshow(kernel)
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


def main():
    parser = argparse.ArgumentParser('Read strip tubes')
    parser.add_argument('--image_file', required=True)
    parser.add_argument('--strip_tl_x', required=True, type=int)
    parser.add_argument('--strip_tl_y', required=True, type=int)
    parser.add_argument('--strip_br_x', required=True, type=int)
    parser.add_argument('--strip_br_y', required=True, type=int)
    parser.add_argument('--strip_count', required=True, type=int)
    parser.add_argument('--plotting', help="Enable plotting", action='store_true')

    args = parser.parse_args()
    results = getPredictions(args.image_file, args.strip_tl_x, args.strip_tl_y, args.strip_br_x,
                             args.strip_br_y, args.strip_count, args.plotting)

    results.append("Control")
    print(results)


if __name__ == '__main__':
    main()

from __future__ import division
import cv2
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt


def getPredictions(image_file, strip_tl_x, strip_tl_y, strip_br_x, strip_br_y, strip_count,
                   plotting):
    image = cv2.imread(image_file)  # image is loaded as BGR

    # Filter the image to enhance various features
    # image = applyClahetoRGB(image, cv2.COLOR_BAYER_BG2RGB)  # Increase contrast to the image

    scores = [None] * strip_count
    strips_tlx = [None] * strip_count  # top left x-values
    strips_tly = [None] * strip_count  # top right x-values
    strips_brx = [None] * strip_count  # bottom right x-values
    strips_bry = [None] * strip_count  # bottom right x-values

    tube_dx = strip_br_x - strip_tl_x
    tube_bottom_height = (strip_br_y - strip_tl_y) / strip_count

    tmp = image.copy()
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
        tmp = cv2.drawContours(tmp, [box], 0, (0, 0, 255), 10)

        # focus in on the tube liquid's enclosing area
        subimage = image[strips_tly[i]:strips_bry[i], strips_tlx[i]:strips_brx[i], 1]
        # # blue channel is all noise, so get rid of it:
        # subimage[:, :, 0] = np.zeros([subimage.shape[0], subimage.shape[1]])

        # We want to get signal from the part of the tube which contains liquid, and not any other
        # background signal. As such, we model the bottom of the tube as a trapezoid and create a
        # kernel to traverse through the tube's enclosing area to find the portion with the highest
        # signal.
        kernel_height = int(tube_bottom_height / 2)
        kernel_width = int(tube_dx / 2)
        kernel = np.ones((kernel_height, kernel_width), np.float32) / (kernel_width * kernel_height)
        # trapezoid = np.zeros((4, 2))
        # trapezoid[0] = np.asarray([0, 0])  # top left
        # trapezoid[1] = np.asarray([tube_dx / 2, 0])  # top right
        # trapezoid[2] = np.asarray([tube_dx / 2, int(tube_bottom_height / 2)])  # bottom right
        # trapezoid[3] = np.asarray([0, int(tube_bottom_height / 2)])  # bottom left
        # cv2.fillPoly(kernel, [trapezoid])
        sums = cv2.filter2D(subimage, -1, kernel)
        _, maxVal, _, _ = cv2.minMaxLoc(sums)

        scores[i] = np.sum(maxVal)

    if plotting:
        fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
    plt.show()

    thresh = scores[-1] * 1.2
    print(scores)

    return ["Positive" if score > thresh else "Negative" for score in scores[0:-1]]


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

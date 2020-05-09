from __future__ import division
import cv2
import numpy as np
import argparse
import scipy
import scipy.stats
import math
from skimage.draw import line, polygon
from sklearn.mixture import GaussianMixture
import glob, os
import json
import imutils
import matplotlib.pyplot as plt
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
from scipy import linalg
import seaborn as sns


sns.set()
sns.set_palette("Paired")
sns.set_style(style='white')
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
plt.rcParams["patch.force_edgecolor"] = False

def getPredictions(image_file, tube_coords_json, plotting, plt_hist=False):
    image = cv2.imread(image_file)  # image is loaded as BGR
    tube_coords = json.loads(tube_coords_json)
    f = open(image_file + ".coords.txt", "w")
    f.write(tube_coords_json)
    f.close()
    im_lower_dim = min(image.shape[0], image.shape[1])
    blur_size = np.round(im_lower_dim / 500)
    blur_size = int(blur_size if blur_size % 2 == 1 else blur_size + 1)
    cv2.GaussianBlur(image, (blur_size, blur_size),
                     cv2.BORDER_DEFAULT)
    # resize large images
    resize_factor = 1
    if im_lower_dim > 1000:
        resize_factor = 1000 / im_lower_dim
        image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor,
                           interpolation=cv2.INTER_AREA)
    i = 1
    tube_coords = np.asarray(tube_coords) * resize_factor
    strip_count = len(tube_coords) - 1
    # Filter the image to enhance various features
    # image = applyClahetoRGB(image, cv2.COLOR_BAYER_BG2RGB)  # Increase contrast to the image

    unstandardized_scores = [None] * strip_count
    unstandardized_scores_medians = [None] * strip_count
    tmp = image.copy()
    tmp_filtered = image.copy()
    sig_dists = []
    sig_coeffs = []

    fig_subplot, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    # iterate over the tubes
    for i in range(0, strip_count):

        tube_width = int(((tube_coords[i][2] - tube_coords[i][0]) ** 2 + (
                tube_coords[i][3] - tube_coords[i][1]) ** 2) ** (1 / 2))
        # let's get background intensity so we can normalize the signal from the fluorescent liquid
        box = np.zeros((5, 2))
        box[0] = np.asarray([tube_coords[i][0], tube_coords[i][1]])  # top right
        box[1] = np.asarray([tube_coords[i + 1][0], tube_coords[i + 1][1]])  # bottom right
        box[2] = np.asarray(extend_line(tube_coords[i + 1][2], tube_coords[i + 1][3],
                                        tube_coords[i + 1][0], tube_coords[i + 1][1],
                                        tube_width / 2.5))  # bottom left
        box[3] = np.asarray(extend_line(tube_coords[i][2], tube_coords[i][3], tube_coords[i][0],
                                        tube_coords[i][1], tube_width / 2.5))  # top left
        box[4] = np.asarray([tube_coords[i][0], tube_coords[i][1]])  # top right
        rr_bg, cc_bg = polygon(box[:, 0], box[:, 1])
        mask_background = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask_background[cc_bg, rr_bg] = 1
        subimage_bg = cv2.bitwise_and(image, image, mask=mask_background)
        # subimage_bg[:, :, 0] = np.zeros([subimage_bg.shape[0], subimage_bg.shape[1]])
        bkgd_red = np.median(
            subimage_bg[cc_bg, rr_bg, 2])  # (np.sum(subimage_bg[:, :, 2])) / np.sum(mask)
        bkgd_grn = np.median(
            subimage_bg[cc_bg, rr_bg, 1])  # (np.sum(subimage_bg[:, :, 1])) / np.sum(mask)
        bkgd_blu = np.median(
            subimage_bg[cc_bg, rr_bg, 0])  # (np.sum(subimage_bg[:, :, 0])) / np.sum(mask)
        b_bg, g_bg, r_bg = cv2.split(subimage_bg)
        hist_begin = 1
        hist_end = 255
        hist_bg, edges_bg = np.histogram(g_bg.ravel(), hist_end - hist_begin,
                                         [hist_begin, hist_end])
        edges_bg = (edges_bg[:-1] + edges_bg[1:]) / 2
        # Sometimes we see bright blue/white artifacts in the image. We have to remove them.
        b_bg, g_bg, r_bg, blue_mask_bg = remove_bright_blues(b_bg, g_bg, r_bg, bkgd_blu, tube_width)
        # blue channel is all noise, so get rid of it:
        b_bg[:, :] = np.zeros([b_bg.shape[0], b_bg.shape[1]])
        # red channel is mostly UTM, so get rid of it:
        r_bg[:, :] = np.zeros([r_bg.shape[0], r_bg.shape[1]])
        # if plotting and i > 3:
        # tmp = cv2.drawContours(tmp, [np.array(box[0:4]).reshape((-1, 1, 2)).astype(np.int32)],
        #                        0, (227, 206, 166), 2)

        # now, define a subimage for the signal in the tube
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
        b, g, r = cv2.split(subimage)

        # Sometimes we see bright blue/white artifacts in the image. We have to remove them.
        b, g, r, blue_mask = remove_bright_blues(b, g, r, bkgd_blu, tube_width)
        # blue channel is all noise, so get rid of it:
        b[:, :] = np.zeros([b.shape[0], b.shape[1]])
        # red channel is mostly UTM, so get rid of it:
        r[:, :] = np.zeros([b.shape[0], b.shape[1]])
        subimage = cv2.merge([b, g, r])

        # subtract away background noise level
        # g_mask = g[:, :] < (bkgd_grn.astype("uint8") + 1)
        # r_mask = r[:, :] < (bkgd_red.astype("uint8") + 1)
        # g -= bkgd_grn.astype("uint8")
        # r -= bkgd_red.astype("uint8")
        # g[g_mask] = 0
        # r[r_mask] = 0
        hist_g, edges_g = np.histogram(g.ravel(), hist_end - hist_begin, [hist_begin, hist_end])
        # g[blue_mask] = int(np.mean(g[cc, rr]))
        # # shift green channel to match with background distribution:
        # green_shift = np.argmax(hist_g) - np.argmax(hist_bg)
        green_shift = 0
        # print("green_shift: {}".format(green_shift))
        # if green_shift > 0:
        #     g_mask = g[:, :] <= (green_shift.astype("uint8"))
        #     g -= green_shift.astype("uint8")
        #     g[g_mask] = 0
        # else:
        #     green_shift *= -1
        #     g_mask = g[:, :] > 255 - (green_shift.astype("uint8"))
        #     g += green_shift.astype("uint8")
        #     g[g_mask] = 255

        # r[blue_mask] = int(np.mean(r[cc, rr])) # red channel is mostly UTM, so get rid of it.
        subimage = cv2.merge([b, g, r])
        # We want to get signal from the part of the tube which contains liquid, and not any other
        # background signal. As such, we model the bottom of the tube as a trapezoid and create a
        # kernel to traverse through the tube's enclosing area to find the portion with the highest
        # signal.
        tube_height = int(((box[3][0] - box[0][0]) ** 2 + (box[3][1] - box[0][1]) ** 2) ** (1 / 2))
        angle = np.rad2deg(np.arctan2(box[0][1] - box[1][1], box[1][0] - box[0][0]))
        kernel = create_kernel(tube_width, tube_height, angle, plotting)
        blurs_green = cv2.filter2D(subimage[:, :, 1], cv2.CV_32F, kernel, anchor=(0, 0))
        blurs_red = cv2.filter2D(subimage[:, :, 2], -1, kernel)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(
            blurs_green)  # + blurs_red) # maxLoc is (x, y) == (col, row)
        max_row = maxLoc[1]
        max_col = maxLoc[0]

        # get list of pixels identified as the tube's signal.
        signal_pxs = []
        for m in range(0, kernel.shape[0]):
            if max_row + m >= g.shape[0]:
                print("Error: please place more space between the strips and the edge of"
                      " the captured image")
                break
            for n in range(0, kernel.shape[1]):
                if max_col + n >= g.shape[1]:
                    print("Error: please place more space between the strips and the edge of"
                          " the captured image")
                    break
                if kernel[m, n] != 0:
                    signal_pxs.append(g[max_row + m, max_col + n])
                    # if plotting and (i == 4 or i == 5):
                    #     # visualize where the kernel is placed
                    #     tmp[max_row + m, max_col + n, 0] = 255
                    #     tmp[max_row + m, max_col + n, 1] = 255
                    #     tmp[max_row + m, max_col + n, 2] = 255

        hist_sig, edges_sig = np.histogram(signal_pxs, hist_end - hist_begin,
                                           [hist_begin, hist_end])

        edges_sig -= green_shift
        edges_sig = edges_sig[0:-1]
        sig_peak = np.max(hist_sig)
        signal_peak_loc = edges_sig[np.argmax(hist_sig)]
        # Make sure background signal within subimage is callibrated to local background signal
        # Fit gaussian for in-tube signal:
        p0 = sig_peak, signal_peak_loc, 4
        coeff = []
        var_matrix = []
        sig_mean = np.mean(signal_pxs)
        # if the image is really bright, it saturates and our curve fitting fails because the center
        # of the gaussian is above 255. In this case, we just set the peak equal to the mean, and
        # based on previous manual inspection of high signals, sd to 20
        if sig_mean > 215:
            coeff = [sig_peak, sig_mean, 20]
        else:
            coeff, var_matrix = curve_fit(gauss, edges_sig, hist_sig, p0=p0)
        # Get the fitted curve
        hist_signal_fit = gauss(edges_sig, *coeff)
        # plt.plot(edges_sig, hist_sig, label='Test data')
        # plt.plot(edges_sig, hist_signal_fit, label='Fitted data')
        # Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
        # print('Fitted mean = ', coeff[1])
        # print('Fitted standard deviation = ', coeff[2])
        bg_peak = np.max(hist_bg)
        bg_peak_loc = edges_bg[np.argmax(hist_bg)]
        p_bg = bg_peak, bg_peak_loc, 4
        coeff_bg, var_matrix_bg = curve_fit(gauss, edges_bg, hist_bg, p0=p_bg)
        # print('Fitted mean = ', coeff_bg[1])
        # print('Fitted standard deviation = ', coeff_bg[2])
        # Get the fitted curve
        hist_bg_fit = gauss(edges_bg, *coeff_bg)
        sig_dist = []
        for j in range(0, len(hist_bg)):
            sig_dist.extend([edges_sig[j]] * hist_sig[j])
        sig_dist -= coeff_bg[1]
        sig_dists.append(sig_dist)
        sig_coeffs.append(coeff)
        # if plotting and plt_hist:
        #     plt.hist(g_bg[cc_bg, rr_bg].ravel(), bins=40, label="background")
        #     plt.hist(sig_dist, bins=40, label="signal")
        #     plt.legend()
        #     plt.title("tube {}".format(i))
        #     plt.show()

        # if plotting and (i == 4 or i == 5):
        #     kernel_cp = kernel > 0
        #     kernel_cp.dtype = 'uint8'
        #     contours, hierarchy = cv2.findContours(kernel_cp, cv2.RETR_EXTERNAL,
        #                                                cv2.CHAIN_APPROX_NONE)
        #     for z in range(0, len(contours[0])):
        #         contours[0][z][0][0] += max_col
        #         contours[0][z][0][1] += max_row
        #
        #     cv2.drawContours(tmp, contours, 0, (255, 255, 255), 2)

        if plotting:
            # if i == strip_count - 1:
            #     tmp = cv2.drawContours(tmp,
            #                            [np.array(box[0:4]).reshape((-1, 1, 2)).astype(np.int32)],
            #                            0, (255, 0, 0), 2)
            # elif i > 3:
            #     tmp = cv2.drawContours(tmp,
            #                            [np.array(box[0:4]).reshape((-1, 1, 2)).astype(np.int32)],
            #                            0, (138, 223, 178), 2)
            # tmp = cv2.circle(tmp, maxLoc, radius=5, color=(255, 255, 255), lineType=cv2.FILLED)
            if plt_hist:
                if i > 3:
                    ax1 = None
                    if i == 4:
                        ax1 = axs[0, 0]
                    elif i == 5:
                        ax1 = axs[0, 1]
                    elif i == 6:
                        ax1 = axs[1, 0]
                    elif i == 7:
                        ax1 = axs[1, 1]
                    # ax1.hist(g.ravel(), hist_end - hist_begin, [hist_begin, hist_end],
                    #          log=True, label="subimage")
                    # sns.distplot(g_bg[cc_bg, rr_bg].ravel())
                    ax1.hist(g_bg[cc_bg, rr_bg].ravel(), hist_end - hist_begin,
                             [hist_begin, hist_end],
                             log=True, label="background")
                    ax1.plot(edges_bg, hist_bg_fit, '--', label='background fit')
                    ax1.hist(signal_pxs, hist_end - hist_begin, [hist_begin, hist_end],
                             log=True, label="signal")
                    ax1.plot(edges_sig, hist_signal_fit, '--', label='signal fit')
                    ax1.set_title('Tube {}'.format(i - 3))

                    ax1.set_ylim([0.5, np.max(hist_bg) * 2])
                    ax1.set_xlim([0, 200])
                    if i == 7:
                        ax1.set_title('Control'.format(i))
                        ax1.legend()
            fig_subplot.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel("Number of pixels")
            plt.ylabel("Pixel intensity")

        unstandardized_scores[i] = coeff[1] - coeff_bg[1]  # np.median(signal_pxs) - coeff_bg[1]
        unstandardized_scores_medians[i] = np.median(signal_pxs) - coeff_bg[1]

        # unstandardized_scores[i] = abs(maxVal - bkgd_grn.astype("uint8"))
        # print("maxVal: {}".format(maxVal))
        # print("bkgd_grn: {}".format(bkgd_grn.astype("uint8")))
        # print("unstandardized_scores: {}".format(unstandardized_scores[i]))
        # print()
        # unstandardized_scores[i] = maxVal

    plt.show()
    if plotting:
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
        plt.title('{}'.format(image_file.split('\\')[-1]))
        plt.show()
    # print("unstandardized_scores: {}".format(unstandardized_scores))
    for i in range(0, len(sig_dists)):
        continue
        # if plotting and plt_hist:
        #     plt.hist(sig_dists[i], bins=40, label="tube {} signal".format(i))
        #     plt.hist(sig_dists[-1], bins=40, label="control signal")
        #     plt.legend()
        #     plt.show()
    # final_score = [unstandardized_score / unstandardized_scores[-1]
    #                for unstandardized_score in unstandardized_scores]
    # print(unstandardized_scores)
    final_score = list((unstandardized_score - unstandardized_scores[-1]) / sig_coeffs[-1][2]
                       for unstandardized_score in unstandardized_scores)
    final_score_medians = list(
        (unstandardized_score_median - unstandardized_scores[-1]) / sig_coeffs[-1][2]
        for unstandardized_score_median in unstandardized_scores_medians)
    # print(sig_coeffs[-1][2])
    f = open(image_file + ".scores.txt", "w")
    f.write(json.dumps(final_score))
    f.close()

    return final_score, final_score_medians


def extend_line(x1, y1, x2, y2, length):
    lenAB = math.sqrt(pow(x1 - x2, 2.0) + pow(y1 - y2, 2.0))
    x = x2 + (x2 - x1) / lenAB * length
    y = y2 + (y2 - y1) / lenAB * length
    return x, y


def gauss(x, *p):
    A, mu, sigma = p
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


def remove_bright_blues(b, g, r, bkgd_blu, tube_width):
    bf = b.astype(float) + .001
    rf = r.astype(float) + .001
    gf = g.astype(float) + .001
    blue_cutoff = 2 * bkgd_blu
    blue_mask1 = bf[:, :] > blue_cutoff
    blue_mask2 = bf[:, :] / gf[:, :] > 0.8
    blue_mask = np.logical_and(blue_mask1, blue_mask2)
    pixel_threshold = (tube_width / 10) ** 2
    if np.sum(blue_mask) > pixel_threshold:
        # be more stringent if we think we've detected a bright blue/white artifact
        blue_cutoff = bkgd_blu * 1.5
        blue_mask = b[:, :] > blue_cutoff
    g[blue_mask] = 0
    r[blue_mask] = 0

    return b, g, r, blue_mask


def create_kernel(tube_width, tube_height, angle, plotting):
    """
    :return: a trapezoidal kernel
    kernel_height = int(tube_height
    """
    kernel_width = tube_width
    kernel_height = tube_height
    kernel = np.zeros((kernel_height, kernel_width), np.float32)
    trap_height_large = int(24 * tube_height / 75)
    trap_height_small = int(11 * tube_height / 75)
    trap_length = int(33 * tube_width / 75)
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


def run_analysis(file, tube_coords, threshold, plotting=True, plt_hist=False):
    fs, fs_m = getPredictions(file, tube_coords, plotting, plt_hist)
    calls = [1 if fs[i] > threshold and fs_m[i] > threshold else 0 for i in range(0, len(fs))]
    print(json.dumps({"calls": calls, "final_scores": fs, "final_scores_median": fs_m}))


def main():
    parser = argparse.ArgumentParser('Read strip tubes')
    parser.add_argument('--image_file', required=False)
    parser.add_argument('--tubeCoords', required=False)
    parser.add_argument('--plotting', help="Enable plotting", action='store_true')
    args = parser.parse_args()
    threshold = 1.5

    # fs = [8.455392751977696, 46.247966797288534, 0.7423274960752811, 0.0]
    fs = [23.404471590657252, 122.53692189214857, 3.1725853678193516, 1.225410678847778]
    data = dict(zip(["Tube 1", "Tube 2", "Tube 3", "Control"], fs))
    tubes = list(data.keys())
    values = list(data.values())
    plt.figure(figsize=(3.8,4.8))
    plt.bar([1,2,3,4], values, width = 0.4, log=True)
    plt.xticks([1,2,3,4], tubes, rotation=60)
    plt.ylabel("Background subtracted fluorescence")
    plt.title("Signal detected per tube")
    plt.hlines(5.15, 0.5, 4.5, colors=["#ffb482"], linestyles='dashed')
    plt.show()

    return

    train_threshold = False
    if train_threshold:
        thresholds = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
    elif args.image_file is None:
        for file in glob.glob(
                r'C:\Users\Sameed\Documents\Educational\PhD\Rotations\Pardis\SHERLOCK-reader\covid\jon_pictures\uploads\*jpg'):
            # files = ["33b", "d1f3", "g.jpg-2020-05-07T170653394Z"]
            files = ["33b"]
            if not any([f in file for f in files]):  # and "mins" not in file:
                continue
            print(file)
            tube_coords = None
            with open(file + ".coords.txt") as f:
                for line in f:  # there should only be one line in file f
                    tube_coords = line
            run_analysis(file, tube_coords, threshold, plotting=True, plt_hist=True)
            print()
    else:
        final_scores = run_analysis(args.image_file, args.tubeCoords, threshold, args.plotting)


if __name__ == '__main__':
    main()

from __future__ import division
import statistics
import cv2
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import StratifiedKFold


class MaxDetector:
    def __init__(self):
        self.positive_signal_values = []
        self.negative_signal_values = []

    def analyze_strip_with_known_truth(self, signal, signal_truth):
        if signal_truth == 'pos':
            self.positive_signal_values.append(self.identify_single_max(signal))
        else:
            self.negative_signal_values.append(self.identify_single_max(signal))

    def identify_single_max(self, data):
        half_data = data[int(data.shape[0] / 2):]
        m = np.max(half_data)
        return m

    def predict_using_single_max(self, data):
        # Standard deviation and mean calculated with emperical estimates
        sd = 12.027
        mean = 54.761
        LODThreshold = mean + 2 * sd
        m = self.identify_single_max(data)
        if m > LODThreshold:
            return "POSITIVE"
        else:
            return "BORDERLINE"

    def print_summary(self):
        pos_mean = np.mean(self.positive_signal_values)
        pos_std = np.std(self.positive_signal_values)
        neg_mean = np.mean(self.negative_signal_values)
        neg_std = np.std(self.negative_signal_values)
        print('#############################')
        print('#### Max Method Summary #####')
        print('#############################')
        print('#### Positive Stats ####')
        print('Mean: {}\nSTD: {}\nFull: {}\n'.format(pos_mean, pos_std,
                                                     sorted(self.positive_signal_values)))
        print('#### Negative Stats ####')
        print('Mean: {}\nSTD: {}\nFull: {}\n'.format(neg_mean, neg_std,
                                                     sorted(self.negative_signal_values)))
        print()


class PeakDetector:
    def __init__(self):
        self.peaks_method_false_neg_no_peak = []
        self.peaks_method_true_pos_detected_peaks = []
        self.peaks_method_true_neg_true_no_peak = []
        self.peaks_method_false_pos_peak_detected = []
        self.true_pos_prominences = []
        self.true_pos_prom_width_ratios = []
        self.true_pos_integral_ratios = []
        self.true_pos_widths = []
        self.false_pos_prominences = []
        self.false_pos_prom_width_ratios = []
        self.false_pos_integral_ratios = []
        self.false_pos_widths = []

    def analyze_strip_with_known_truth(self, full_signal, image_name, signal_truth):
        peaks_method_ret = self.find_peaks(full_signal, image_name)

        # add this result to aggregate statistics
        if signal_truth == 'pos':
            if peaks_method_ret == 'NO PEAK':
                self.peaks_method_false_neg_no_peak.append(image_name)
                return 'NO PEAK'
            else:
                self.peaks_method_true_pos_detected_peaks.append(peaks_method_ret)
        else:
            if peaks_method_ret == 'NO PEAK':
                self.peaks_method_true_neg_true_no_peak.append(image_name)
                return 'NO PEAK'
            else:
                self.peaks_method_false_pos_peak_detected.append(peaks_method_ret)

        prominence, prom_width_ratio, integral_ratio, width = peaks_method_ret

        # if a short peak is super wide compared to its prominence, the peak tends to come from a
        # reflection of light from tape on the lateral flow strip which has not been fully pat down.
        if width > 65 and prom_width_ratio < 0.65:
            return 'INCONCLUSIVE - check strip tape'
        if signal_truth == 'pos':
            self.true_pos_prominences.append(prominence)
            self.true_pos_prom_width_ratios.append(prom_width_ratio)
            self.true_pos_integral_ratios.append(integral_ratio)
            self.true_pos_widths.append(width)
        else:
            self.false_pos_prominences.append(prominence)
            self.false_pos_prom_width_ratios.append(prom_width_ratio)
            self.false_pos_integral_ratios.append(integral_ratio)
            self.false_pos_widths.append(width)

    def find_peaks(self, full_signal, image_name):
        """
        use peak detection to find peaks based on prominence and relative height, width.
        return:
            NEGATIVE if no signal peak detected
            else:
            prominence of signal peak, prominence/width ratio, ratio of integral of left side of
                peak to right side of the peak, and width of the peak.
        """

        # We only need to work on the second half of the data; this will contain our peaks.
        signal = full_signal[int(full_signal.shape[0] / 2):]
        peaks, _ = find_peaks(signal, height=20, prominence=10)
        peaks = sorted(peaks, key=lambda peak: signal[peak], reverse=True)
        prominences, left_bases, right_bases = peak_prominences(signal, peaks)
        if len(prominences) == 0:
            return "NO PEAK"
        results_full = peak_widths(signal, peaks, rel_height=0.75)
        left_side_sum = 0
        right_side_sum = 0
        for i in range(int(results_full[2][0]), peaks[0]):
            left_side_sum += signal[i]
        for i in range(peaks[0] + 1, int(results_full[3][0])):
            right_side_sum += signal[i]

        # fig, ax = plt.subplots()
        # ax.plot(full_signal, 'b')
        # ax.set_title(image_name)
        # ax.plot([int(full_signal.shape[0] / 2) + peak for peak in
        #          peaks], signal[peaks], "x", color="orange", markersize=30)
        # fig.savefig('{}_peaks.png'.format(image_name))
        # plt.close(fig)

        # return prominence, prominence/width ratio, and ratio of integral
        return prominences[0], prominences[0] / results_full[0][0], left_side_sum / (
                left_side_sum + right_side_sum), results_full[3][0] - results_full[2][0]

    def print_summary(self):
        print('###############################')
        print('#### Peaks Method Summary #####')
        print('###############################')
        print('True positives: {}'.format(len(self.peaks_method_true_pos_detected_peaks)))
        print('False Negatives: {}'.format(self.peaks_method_false_neg_no_peak))
        print('True Negative: {}'.format(self.peaks_method_true_neg_true_no_peak))
        print('False positives: {}'.format(len(self.peaks_method_false_pos_peak_detected)))

        print("true_pos_prominences mean: {}; sd: {}".format(np.mean(self.true_pos_prominences),
                                                             np.std(self.true_pos_prominences)))
        print("true_pos_prom_width_ratios mean: {}; sd: {}".format(
            np.mean(self.true_pos_prom_width_ratios), np.std(self.true_pos_prom_width_ratios)))
        print("true_pos_integral_ratios mean: {}; sd: {}".format(
            np.mean(self.true_pos_integral_ratios), np.std(self.true_pos_integral_ratios)))
        print("true_pos_widths mean: {}; sd: {}".format(np.mean(self.true_pos_widths),
                                                        np.std(self.true_pos_widths)))
        print("false_pos_prominences mean: {}; sd: {}".format(np.mean(self.false_pos_prominences),
                                                              np.std(self.false_pos_prominences)))
        print("false_pos_prom_width_ratios mean: {}; sd: {}".format(
            np.mean(self.false_pos_prom_width_ratios), np.std(self.false_pos_prom_width_ratios)))
        print("false_pos_integral_ratios mean: {}; sd: {}".format(
            np.mean(self.false_pos_integral_ratios), np.std(self.false_pos_integral_ratios)))
        print("false_pos_widths mean: {}; sd: {}".format(np.mean(self.false_pos_widths),
                                                         np.std(self.false_pos_widths)))
        print()

        bins = np.linspace(0, 250, 100)
        fig, ax = plt.subplots()
        ax.hist(self.true_pos_prominences, bins, alpha=0.5, label='true pos')
        ax.hist(self.false_pos_prominences, bins, alpha=0.5, label='false pos')
        ax.set_title('Prominences')
        ax.set_xlabel('Peak prominence')
        ax.set_ylabel('Count')
        ax.legend(loc='upper right')
        fig.savefig('prominences_histogram.png')
        plt.close(fig)

        bins = np.linspace(0, 4, 80)
        fig, ax = plt.subplots()
        ax.hist(self.true_pos_prom_width_ratios, bins, alpha=0.5, label='true pos')
        ax.hist(self.false_pos_prom_width_ratios, bins, alpha=0.5, label='false pos')
        ax.set_title('Peak prominence divided by peak width')
        ax.set_xlabel('[Peak prominence]/[peak width]')
        ax.set_ylabel('Count')
        ax.legend(loc='upper right')
        fig.savefig('prominence_to_width_ratios_histogram.png')
        plt.close(fig)

        bins = np.linspace(0, 1, 25)
        fig, ax = plt.subplots()
        ax.hist(self.true_pos_integral_ratios, bins, alpha=0.5, label='true pos')
        ax.hist(self.false_pos_integral_ratios, bins, alpha=0.5, label='false pos')
        ax.set_title('Relative area under left half of peaks')
        ax.set_xlabel('Area under left half of peak')
        ax.set_ylabel('Count')
        ax.legend(loc='upper right')
        fig.savefig('integral_ratio_histograms.png')
        plt.close(fig)

        bins = np.linspace(0, 100, 50)
        fig, ax = plt.subplots()
        ax.hist(self.true_pos_widths, bins, alpha=0.5, label='true pos')
        ax.hist(self.false_pos_widths, bins, alpha=0.5, label='false pos')
        ax.set_title('Peak widths')
        ax.set_xlabel('Peak width')
        ax.set_ylabel('Count')
        ax.legend(loc='upper right')
        fig.savefig('widths_histogram.png')
        plt.close(fig)


class NeuralNetDetector:
    def __init__(self, input_dims):
        self.input_dims = input_dims

    def analyze_cross_validated_performance(self, data, labels):
        n_folds = 10
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
        i = 0
        for train, test in skf.split(data, labels):
            print("Running Fold", i + 1, "/", n_folds)
            i += 1
            model = None  # Clearing the NN.
            model = self.create_model()
            self.train_and_evaluate_model(model, data[train], labels[train], data[test],
                                          labels[test])

    def create_model(self):
        model = Sequential()
        model.add(Dense(96, activation='relu', input_dim=self.input_dims))
        model.add(Dense(32, activation='relu', input_dim=self.input_dims))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_and_evaluate_model(self, model, data_train, labels_train, data_test, labels_test):
        model.fit(data_train, labels_train, epochs=10, batch_size=32)
        score = model.evaluate(data_test, labels_test, batch_size=128)
        print(score)


def correct_input_image(nimg, correction_method):
    hsv = cv2.cvtColor(nimg, cv2.COLOR_BGR2HSV)
    # determine the hsv color of this strip
    if correction_method == 'hsv_value':
        cimg = hsv_value_correction(hsv)
    elif correction_method == 'clahe':
        cimg = clahe_correction(hsv)
    elif correction_method == 'hist_eq':
        cimg = histogram_equalization_correction(hsv)
    elif correction_method == 'gray':
        cimg = gray_correction(hsv)
    else:
        raise Exception('incorrect normalization type')
    return cimg


def hsv_value_correction(hsv):
    # We chose a standard hsv value to normalize all of our images to:
    standard_v = 240.

    # First, we calculate the median hsv V value of the image as the median value of the
    # median V values from each row in the image:
    rows_v = []
    for i in range(hsv.shape[0]):
        pixels_v = np.zeros(hsv.shape[1])
        for j in range(hsv.shape[1]):
            pixels_v[j] = hsv[i, j][2]
        rows_v.append(statistics.median(pixels_v))
        # print(rows_hsv)
    rows_v = np.array(rows_v)
    strip_v = statistics.median(rows_v)
    val_correction = standard_v / strip_v
    for i in range(hsv.shape[0]):
        for j in range(hsv.shape[1]):
            hsv[i, j][2] = min(255, hsv[i, j][2] * val_correction)

    return cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)


def clahe_correction(hsv):
    # We normalize image brightness and then apply clahe
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10, 10))
    return clahe.apply(hsv_value_correction(hsv))


def histogram_equalization_correction(hsv):
    # We normalize image brightness and then apply clahe
    return cv2.equalizeHist(hsv_value_correction(hsv))


def gray_correction(hsv):
    # Not really a correction - just changes signal to gray
    return cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)


def convert_image_to_linear_signal(cimg):
    # Convert the extracted, corrected image to a linear signal.
    data = cimg.astype('int32')
    nrows = data.shape[0]
    signal = np.array([None] * nrows)
    for r in range(0, nrows):
        signal[r] = 255 - np.mean(data[r])
    return signal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--glob_path', required=True,
                        help='The path that strip_analysis should search'
                             'for matching files in. \nThis input allows'
                             'Unix-style globbing, e.g. "images/*raw*.png'
                             'will search for all images matching the '
                             'pattern described. Alternatively, you can'
                             'also just stipulate something like'
                             'images/raw17.png')
    parser.add_argument('--correction_method',
                        choices=['hsv_value', 'clahe', 'gray', 'hist_eq'],
                        help='choice of normalization method for the raw image', default='clahe')
    args = parser.parse_args()
    glob_path = args.glob_path
    correction_method = args.correction_method

    # get all the images which match the input path
    image_files = glob.glob(glob_path)

    peak_detector = PeakDetector()
    max_detector = MaxDetector()
    signals = []
    labels = []
    for i, image_file in enumerate(image_files):
        nimg = cv2.imread(image_file)
        # The image name should be in the format {description}_{pos/neg}.{format}, e.g.
        # N2-for-LOD_raw_strip10_pos.png
        file_name = image_file.strip().split('.')
        image_name = file_name[0].split('/')[-1]
        signal_truth = file_name[0].split('_')[-1]
        if signal_truth != 'pos' and signal_truth != 'neg':
            raise Exception('The image file name should be in the format'
                            '{description}_{pos/neg}.{format}'
                            'e.g. N2-for-LOD_raw_strip10_pos.png')

        signal = convert_image_to_linear_signal(correct_input_image(nimg, correction_method))
        # for NN
        labels.append(signal_truth == 'pos')
        signals.append(signal)
        max_detector.analyze_strip_with_known_truth(signal, signal_truth)
        peak_detector.analyze_strip_with_known_truth(signal, image_name, signal_truth)

    print('total positive strips: {}'.format(labels.count(True)))
    print('total negative strips: {}\n\n'.format(labels.count(False)))

    max_detector.print_summary()
    peak_detector.print_summary()

    # nn = NeuralNetDetector(len(signals[0]))
    # nn.analyze_cross_validated_performance(np.array(signals), np.array(labels))

    # Using the means and the standard deviations calculated above, we can use a prediction
    # algorithm to see how our data does.


if __name__ == '__main__':
    main()

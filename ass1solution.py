import math
import numpy as np
from scipy.signal import medfilt
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from scipy.io.wavfile import read
import glob
import os

########################### A. Block-wise Pitch Tracking with the ACF ###########################

def block_audio(x, blockSize, hopSize, fs):
    i = 0
    xb = []
    timeInSec = []
    while i < len(x):
        timeInSec.append(i / fs)
        chunk = x[i: i + blockSize]
        if len(chunk) != blockSize:
            chunk = np.append(chunk, np.zeros(blockSize - len(chunk)))
            xb.append(chunk)
            break
        else:
            xb.append(chunk)
        i += hopSize

    return [np.array(xb), np.array(timeInSec)]
    
def comp_acf(inputVector, bIsNormalized=True):
    r = np.correlate(inputVector, inputVector, 'full')
    if bIsNormalized:
        r = r/(np.sum(np.square(r)))
    return r[len(r) // 2 :]

def get_f0_from_acf(r, fs):
    peaks = find_peaks(r, height=0, distance=50)[0]
    if len(peaks) >= 2:
        p = sorted(r[peaks])[::-1]
        sorted_arg = np.argsort(r[peaks])[::-1]
        f0 = fs/abs(peaks[sorted_arg][1] - peaks[sorted_arg][0])
        # plt.plot(r)
        # plt.plot(peaks, r[peaks], 'rs')
        # plt.show()
        return f0
    return 0

def track_pitch_acf(x, blockSize, hopSize, fs):
    blocked_x, timeInSec = block_audio(x, blockSize, hopSize, fs)
    frequencies = []
    for b in blocked_x:
        acf = comp_acf(b)
        f0 = get_f0_from_acf(acf, fs)
        frequencies.append(f0)
    return [np.array(frequencies), timeInSec]

########################### B. Evaluation ###########################

def gen_sin(f1=441, f2=882, fs=44100):
    t1 = np.linspace(0, 1, fs)
    t2 = np.linspace(1, 2, fs)
    sin_441 = np.sin(2 * np.pi * 441 * t1)
    sin_882 = np.sin(2 * np.pi * 882 * t2)
    sin = np.append(sin_441, sin_882)
    return sin

def code_for_B1():
    fs = 44100
    f1 = 441
    f2 = 882
    sin = gen_sin(f1, f2, fs)
    [frequencies, timeInSec] = track_pitch_acf(sin, 441, 441, fs)
    error = np.zeros(len(timeInSec))
    error[:len(timeInSec) // 2] += f1
    error[len(timeInSec) // 2 :] += f2
    error = np.abs(error - frequencies)
    plt.plot(timeInSec, error)
    plt.plot(timeInSec, frequencies)
    plt.show()

code_for_B1()

def convert_freq2midi(freqInHz):
    return 69 + 12 * np.log2(freqInHz / 440.0)

def freq2cent(freqInHz):
    return 1200 * np.log2(freqInHz / 440.0)

def eval_pitchtrack(estimateInHz, groundtruthInHz):
    centError = []
    for i in range(len(groundtruthInHz)):
        if groundtruthInHz[i] != 0:
            if estimateInHz[i] != 0:
                centError.append(freq2cent(estimateInHz[i]) - freq2cent(groundtruthInHz[i]))
            elif estimateInHz[i] == 0:
                centError.append(-freq2cent(groundtruthInHz[i]))
    centError = np.array(centError)
    rms = np.sqrt(np.mean(np.square(centError)))
    return rms

def run_evaluation(complete_path_to_data_folder):
    file_path = os.path.join(complete_path_to_data_folder, '*.wav')
    wav_files = [f for f in glob.glob(file_path)]
    errCentRms = []
    for wav_file in wav_files:
        name = os.path.split(wav_file)[1].split('.')[0]
        # name = wav_file.split('/')[-1].split('.')[0]
        txt_file = os.path.join(complete_path_to_data_folder, name+'.f0.Corrected.txt')
        with open(txt_file) as f:
            annotations = f.readlines()
        for i in range(len(annotations)):
            annotations[i] = list(map(float, annotations[i][:-2].split('     ')))
        annotations = np.array(annotations)
        fs, audio = read(wav_file)
        freq, timeInSec = track_pitch_acf(audio, 1024, 512, fs)
        trimmed_freq = np.ones(freq.shape)
        trimmed_annotations = np.ones(freq.shape)
        for i in range(len(freq)):
            if annotations[i, 2] > 0:
                trimmed_freq[i] = freq[i]
                trimmed_annotations[i] = annotations[i, 2]
        # plt.plot(trimmed_freq)
        # plt.plot(trimmed_annotations)
        # plt.show()
        errCentRms.append(eval_pitchtrack(trimmed_freq, trimmed_annotations))
    errCentRms = np.array(errCentRms)
    # print(errCentRms)
    return np.mean(errCentRms)

########################### C. Bonus ###########################

def get_f0_from_acfmod(r, fs):
    peaks = find_peaks(r, height=0, distance=50)[0]
    if len(peaks) >= 2:
        # p = sorted(r[peaks])[::-1]
        sorted_arg = np.argsort(r[peaks])[::-1]
        px, py = parabolic(r,peaks[sorted_arg][0])
        return fs/px
        # f0 = fs/abs(peaks[sorted_arg][1] - peaks[sorted_arg][0])
        # return f0
    return 0

def track_pitch_acfmod(x, blockSize, hopSize, fs):
    blocked_x, timeInSec = block_audio(x, blockSize, hopSize, fs)
    frequencies = []
    for b in blocked_x:
        acf = comp_acf(b)
        f0 = get_f0_from_acfmod(acf, fs)
        frequencies.append(f0)
    frequencies = np.array(frequencies)
    frequencies = medfilt(frequencies,kernel_size=9)
    return [frequencies, timeInSec]

def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    f is a vector and x is an index for that vector.
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

# fs = 44100
# f1 = 441
# f2 = 882
# sin = gen_sin(f1, f2, fs)
# [frequencies, timeInSec] = track_pitch_acfmod(sin, 441, 441, fs)
# error = np.zeros(len(timeInSec))
# error[:len(timeInSec) // 2] += f1
# error[len(timeInSec) // 2 :] += f2
# error = np.abs(error - frequencies)
# plt.plot(timeInSec, error)
# plt.plot(timeInSec, frequencies)
# plt.show()

def run_evaluation_mod(complete_path_to_data_folder):
    if complete_path_to_data_folder[-1] == '/':
        complete_path_to_data_folder = complete_path_to_data_folder[:-1]
    wav_files = [f for f in glob.glob(complete_path_to_data_folder + '/*.wav')]
    errCentRms = []
    for wav_file in wav_files:
        name = wav_file.split('\\')[-1].split('.')[0]
        with open(complete_path_to_data_folder + '/' + name + '.f0.Corrected.txt') as f:
            annotations = f.readlines()
        for i in range(len(annotations)):
            annotations[i] = list(map(float, annotations[i][:-2].split('     ')))
        annotations = np.array(annotations)
        fs, audio = read(wav_file)
        freq, timeInSec = track_pitch_acfmod(audio, 2048, 512, fs)
        trimmed_freq = np.ones(freq.shape)
        trimmed_annotations = np.ones(freq.shape)
        for i in range(len(freq)):
            if annotations[i, 2] > 0:
                trimmed_freq[i] = freq[i]
                trimmed_annotations[i] = annotations[i, 2]
        # plt.plot(trimmed_freq, label='trimmed frequency')
        # plt.plot(trimmed_annotations, label='trimmed annotation')
        # plt.legend()
        # plt.show()
        errCentRms.append(eval_pitchtrack(trimmed_freq, trimmed_annotations))
    errCentRms = np.array(errCentRms)
    # print(errCentRms)
    return np.mean(errCentRms)

print(run_evaluation("trainData/"))
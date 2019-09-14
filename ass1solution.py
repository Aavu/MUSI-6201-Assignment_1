import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt

# A. Block-wise Pitch Tracking with the ACF

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

    return [np.array(xb),np.array(timeInSec)]

def comp_acf(inputVector, bIsNormalized=True):
    r = np.correlate(inputVector, inputVector, 'full')
    if bIsNormalized:
        r /= (np.max(r) + 1e-6)
    return r[len(r)//2:]

def get_f0_from_acf(r, fs):
    for i in range(1, len(r)-1):
        if r[i-1] < r[i] and r[i] >= r[i+1]:
            return fs/i
    return 0

def track_pitch_acf(x, blockSize, hopSize, fs):
    blocked_x, timeInSec = block_audio(x, blockSize, hopSize, fs)
    frequencies = []
    for b in blocked_x:
        acf = comp_acf(b)
        f0 = get_f0_from_acf(acf, fs)
        frequencies.append(f0)
    return [np.array(frequencies), timeInSec]

def gen_sin(f1=441, f2=882, fs=44100):
    t1 = np.linspace(0, 1, fs)
    t2 = np.linspace(1, 2, fs)
    sin_441 = np.sin(2 * np.pi * 441 * t1)
    sin_882 = np.sin(2 * np.pi * 882 * t2)
    sin = np.append(sin_441, sin_882)
    return sin

# fs = 44100
# f1 = 441
# f2 = 882
# sin = gen_sin(f1, f2, fs)
# [frequencies, timeInSec] = track_pitch_acf(sin, 441, 441, fs)
# error = np.zeros(len(timeInSec))
# error[:len(timeInSec) // 2] += f1
# error[len(timeInSec) // 2 :] += f2
# error = np.abs(error - frequencies)
# plt.plot(timeInSec, error)
# plt.plot(timeInSec, frequencies)
# plt.show()

def convert_freq2midi(freqInHz):
    return 69 + 12*np.log2(freqInHz / 440.0)

def eval_pitchtrack(estimateInHz, groundtruthInHz):
    return np.sqrt(np.mean(np.square(estimateInHz-groundtruthInHz)))

# midi = convert_freq2midi(np.array([440, 880, 660, 293.5]))
# print(midi)

# print(eval_pitchtrack(np.zeros(10), np.array([10,20,30,40,50,60,70,80,90,0])))

def run_evaluation(complete_path_to_data_folder):
    fs, audio = read(complete_path_to_data_folder + '/01-D_AMairena.wav')
    blocked_x, timeInSec = block_audio(audio, 528, 528, fs)
    print(np.round(timeInSec[:10], 3))

run_evaluation("../trainData")
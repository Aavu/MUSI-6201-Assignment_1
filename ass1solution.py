import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import glob

np.set_printoptions(precision=3, suppress=True)
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
        r = r/(np.max(r) + 1e-6)
    return r[len(r)//2:]

def get_f0_from_acf(r, fs):
    for i in range(1, len(r)-1):
        if r[i-1] < r[i] and r[i] >= r[i+1]:
            f0 = fs / i
            # if f0 > 600:
            #     plt.plot(r)
            #     plt.plot([0,i],[r[0],r[i]], 'rs')
            #     plt.show()
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
    if freqInHz == 0:
        return 0
    return 69 + 12*np.log2(freqInHz / 440.0)

def freq2cent(freqInHz):
    if freqInHz == 0:
        return 0
    return 1200 * np.log2(freqInHz/440.0)

def eval_pitchtrack(estimateInHz, groundtruthInHz):
    # return np.sqrt(np.mean(np.square(estimateInHz-groundtruthInHz)))
    centError = []
    for i in range(len(groundtruthInHz)):
        if  groundtruthInHz[i] != 0:
            centError.append(freq2cent(estimateInHz[i]) - freq2cent(groundtruthInHz[i]))
    centError = np.array(centError)
    rms = np.sqrt(np.mean(np.square(centError)))
    # centRms = np.sqrt(np.mean(np.square(convert_freq2midi(estimateInHz)-convert_freq2midi(groundtruthInHz))))
    return rms

def run_evaluation(complete_path_to_data_folder):
    if complete_path_to_data_folder[-1] == '/':
        complete_path_to_data_folder = complete_path_to_data_folder[:-1]
    wav_files = [f for f in glob.glob(complete_path_to_data_folder + '/*.wav')]
    errCentRms = []
    for wav_file in wav_files:
        name = wav_file.split('/')[-1].split('.')[0]
        with open(complete_path_to_data_folder + '/' + name + '.f0.Corrected.txt') as f:
            annotations = f.readlines()
        for i in range(len(annotations)):
            annotations[i] = list(map(float, annotations[i][:-2].split('     ')))
        annotations = np.array(annotations)
        fs, audio = read(wav_file)
        freq, timeInSec = track_pitch_acf(audio, 2048, 512, fs)
        trimmed_freq = np.ones(freq.shape)
        trimmed_annotations = np.ones(freq.shape)
        for i in range(len(freq)):
            if annotations[i, 3] > 0:
                trimmed_freq[i] = freq[i]
                trimmed_annotations[i] = annotations[i, 3]
        plt.plot(trimmed_freq)
        plt.plot(trimmed_annotations)
        plt.show()
        errCentRms.append(eval_pitchtrack(trimmed_freq, trimmed_annotations))
    return errCentRms

print(run_evaluation("trainData"))
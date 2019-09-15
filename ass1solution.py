import math
import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from scipy.io.wavfile import read

def convert_freq2midi(freqInHz):
    pitchInMIDI = 69 + 12 * np.log2(np.divide(freqInHz,440))

    return pitchInMIDI

def eval_pitchtrack(estimateInHz, groundtruthInHz):
    estimateInMIDI = convert_freq2midi(estimateInHz)
    groundtruthInMIDI = convert_freq2midi(groundtruthInHz)
    errCent = np.subtract(estimateInMIDI,groundtruthInMIDI)
    # n = errCent.shape[0] * errCent.shape[1]
    errCentRms = np.sqrt(np.mean(np.square(errCent)))
    # errCentRms = np.sqrt(np.divide(np.sum(np.square(errCent)),n))

    return errCentRms


# Bonus (Code modified from Raghav's version)

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
        if r[i-1] < r[i] and r[i] >= r[i+1]:    # find the peak
            px, py = parabolic(r, i)
            return fs/px
            # return fs/i
    return 0

def track_pitch_acfmod(x, blockSize, hopSize, fs):
    blocked_x, timeInSec = block_audio(x, blockSize, hopSize, fs)
    frequencies = []
    for b in blocked_x:
        acf = comp_acf(b)
        f0 = get_f0_from_acf(acf, fs)
        frequencies.append(f0)
    frequencies = np.array(frequencies)
    frequencies = medfilt(frequencies,kernel_size=5)
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

def gen_sin(f1=441, f2=882, fs=44100):
    t1 = np.linspace(0, 1, fs)
    t2 = np.linspace(1, 2, fs)
    sin_441 = np.sin(2 * np.pi * 441 * t1)
    sin_882 = np.sin(2 * np.pi * 882 * t2)
    sin = np.append(sin_441, sin_882)
    return sin

fs = 44100
f1 = 441
f2 = 882
sin = gen_sin(f1, f2, fs)
[frequencies, timeInSec] = track_pitch_acfmod(sin, 441, 441, fs)
error = np.zeros(len(timeInSec))
error[:len(timeInSec) // 2] += f1
error[len(timeInSec) // 2 :] += f2
error = np.abs(error - frequencies)
plt.plot(timeInSec, error)
plt.plot(timeInSec, frequencies)
plt.show()
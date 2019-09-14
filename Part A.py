import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def block_audio(x,blockSize,hopSize,fs):
    # Reading the audio file

    xb = []
    timeInSec = []
    n = len(x)
    m = (n - hopSize) % (blockSize - hopSize)

    if m > 0:
      x = np.pad(x, (0,m), 'constant', constant_values=0)

    # Blocking the audio file
    for i in range(0, n-blockSize, hopSize):
        time_stamp = i / fs
        block = []
        for j in range(i, i+blockSize):
            block.append(x[j])
        xb.append(block)
        timeInSec.append(time_stamp)

    return xb, timeInSec

def comp_acf(inputVector, bIsNormalized):

    acf = np.correlate(inputVector, inputVector, 'full')

    idx = int(len(acf)//2)
    r = acf[idx:]

    if not bIsNormalized:
        if r.dtype == 'float32':
            audio = r
        else:
            # change range to [-1,1)
            if r.dtype == 'uint8':
                nbits = 8
            elif r.dtype == 'int16':
                nbits = 16
            elif r.dtype == 'int32':
                nbits = 32

            audio = r / float(2 ** (nbits - 1))

        # special case of unsigned format
        if r.dtype == 'uint8':
            audio = audio - 1.

        r = audio

    return r


def get_f0_from_acf(r, fs):

    d = np.diff(r)
    start = np.nonzero(d > 0)[0][0]

    peak = np.argmax(r[start:]) + start

    f0 = fs / peak

    return f0

def track_pitch_acf(x,blockSize,hopSize,fs):

    audio, timeInSec = block_audio(x, blockSize, hopSize, fs)
    acf = comp_acf(audio[6], False)
    f0 = get_f0_from_acf(acf, fs)

    return f0, timeInSec

sample_rate, audio = wav.read('D:/GT_Sem1/GT_Music Information Retrieval/Repo/Assignment1/02.wav')
print(track_pitch_acf(audio, 256, 16, sample_rate))

import math
import numpy as np
from matplotlib import pyplot as plt

def convert_freq2midi(freqInHz):
    pitchInMIDI = 69 + 12 * np.log2(np.divide(freqInHz,440))

    return pitchInMIDI

def eval_pitchtrack(estimateInHz, groundtruthInHz):
    estimateInMIDI = convert_freq2midi(estimateInHz)
    groundtruthInMIDI = convert_freq2midi(groundtruthInHz)
    errCent = np.subtract(estimateInMIDI,groundtruthInMIDI)
    n = errCent.shape[0] * errCent.shape[1]
    errCentRms = np.sqrt(np.divide(np.sum(np.square(errCent)),n))

    return errCentRms
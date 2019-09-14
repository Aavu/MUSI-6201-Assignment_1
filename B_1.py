import math
import numpy as np
from matplotlib import pyplot as plt

# generate test signal
fs = 44100
test_signal = np.zeros((1,2*fs))
for index in range(0,2*fs):
    t = index/fs
    if t>=0 and t<=1:
        test_signal[index] = math.sin(2*math.pi*441*t)
    else:
        test_signal[index] = math.sin(2*math.pi*882*t)

# call the function
blockSize = 
hopSize = 
f0, timeInSec = track_pitch_acf(test_signal,blockSize,hopSize,fs)

error = np.zeros((1,2*fs))
for index in range(0,2*fs):
    t = index/fs
    if t>=0 and t<=1:
        error[index] = f0[index]-441
    else:
        error[index] = f0[index]-882

# plotting
fig, (ax1,ax2) = plt.subplots(2)
ax1.plot(f0)
ax1.set(xlabel='sample', ylabel='f_0 / Hz', title='fundamental frequency')
ax2.plot(error)
ax2.set(xlabel='sample', ylabel='error / Hz', title='error')
plt.show()
import numpy as np
from scipy.signal import lfilter

def get_ssc_feat(x, deadzone, winsize=None, wininc=None, datawin=None, dispstatus=0):
    if winsize is None:
        winsize = x.shape[0]
    if wininc is None:
        wininc = winsize
    if datawin is None:
        datawin = np.ones(winsize)
    
    x = np.vstack([np.zeros((1, x.shape[1])), np.diff(x, axis=0)])

    datasize = x.shape[0]
    Nsignals = x.shape[1]
    numwin = (datasize - winsize) // wininc + 1

    feat = np.zeros((numwin, Nsignals))

    st = 0
    en = winsize

    for i in range(numwin):
        y = x[st:en, :] * datawin[:, None]
        y = (y > deadzone) - (y < -deadzone)

        b = np.exp(-np.arange(1, winsize // 2 + 1))
        z = lfilter(b, 1, y, axis=0)
        z = (z > 0) - (z < -0)
        dz = np.diff(z, axis=0)

        feat[i, :] = np.sum(np.abs(dz) == 2, axis=0)

        st += wininc
        en += wininc

    return feat

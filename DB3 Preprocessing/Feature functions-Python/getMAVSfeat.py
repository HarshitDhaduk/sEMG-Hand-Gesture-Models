import numpy as np

def get_mavs_feat(x, winsize=None, wininc=None, datawin=None, dispstatus=0):
    if winsize is None:
        winsize = x.shape[0]
    if wininc is None:
        wininc = winsize
    if datawin is None:
        datawin = np.ones(winsize)
    
    datasize = x.shape[0]
    Nsignals = x.shape[1]
    numwin = (datasize - winsize) // wininc + 1

    feat = np.zeros((numwin, Nsignals))

    st = 0
    en = winsize

    for i in range(numwin - 1):
        curwin = x[st:en, :] * datawin[:, None]
        curwinSucc = x[st + wininc:en + wininc, :] * datawin[:, None]
        feat[i, :] = np.mean(np.abs(curwinSucc), axis=0) - np.mean(np.abs(curwin), axis=0)
        st += wininc
        en += wininc

    return feat

import numpy as np

def get_hist_feat(x, winsize=None, wininc=None, edges=None):
    if winsize is None:
        winsize = x.shape[0]
    if wininc is None:
        wininc = winsize
    if edges is None:
        edges = np.arange(-3, 3.3, 0.3)

    datasize = x.shape[0]
    Nsignals = x.shape[1]
    numwin = (datasize - winsize) // wininc + 1

    feat = np.zeros((numwin, Nsignals * len(edges)))

    st = 0
    en = winsize

    for i in range(numwin):
        curwin = x[st:en, :]
        F0 = np.array([np.histogram(curwin[:, j], bins=edges)[0] for j in range(Nsignals)]).T
        feat[i, :] = F0.flatten()
        st += wininc
        en += wininc

    return feat

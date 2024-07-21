import numpy as np
import pywt

def get_mdwt_feat(x, winsize=None, wininc=None):
    if winsize is None:
        winsize = x.shape[0]
    if wininc is None:
        wininc = winsize

    datasize = x.shape[0]
    Nsignals = x.shape[1]
    numwin = (datasize - winsize) // wininc + 1

    feat = []

    st = 0
    en = winsize

    for i in range(numwin):
        curwin = x[st:en, :]
        m_xk = []
        for colInd in range(curwin.shape[1]):
            C, L = pywt.wavedec(curwin[:, colInd], 'db7', level=3)
            L = np.cumsum(L)
            L = np.insert(L, 0, 0)

            sReal = [0, 3, 2, 1]

            for s in sReal:
                d_xk = C[L[s]:L[s+1]]
                MaxSum = min(int(np.ceil(x.shape[0] / (2**sReal[s] - 1))), len(d_xk))
                m_xk.append(np.sum(np.abs(d_xk[:MaxSum])))

        feat.append(np.array(m_xk).flatten())

        st += wininc
        en += wininc

    return np.array(feat)

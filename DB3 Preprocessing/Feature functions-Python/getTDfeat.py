import numpy as np

def get_td_feat(x, deadzone=1e-5, winsize=None, wininc=None):
    if winsize is None:
        winsize = x.shape[0]
    if wininc is None:
        wininc = winsize
    
    # Compute individual features
    feat1 = get_mav_feat(x, winsize, wininc)
    feat2 = get_mavs_feat(x, winsize, wininc)
    feat3 = get_zc_feat(x, deadzone, winsize, wininc)
    feat4 = get_ssc_feat(x, deadzone, winsize, wininc)
    feat5 = get_wl_feat(x, winsize, wininc)

    # Combine features into a single matrix
    feat = np.hstack([feat1, feat2, feat3, feat4, feat5])

    return feat

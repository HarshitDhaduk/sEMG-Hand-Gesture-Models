import os
import numpy as np
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor
import logging

# Importing the feature extraction functions
from getRMSfeat import get_rms_feat
from getIAVfeat import get_iav_feat
from getMAVfeat import get_mav_feat
from getMAVSfeat import get_mavs_feat
from getSSCfeat import get_ssc_feat
from getWLfeat import get_wl_feat
from getZCfeat import get_zc_feat
from getTDfeat import get_td_feat
from getHISTfeat import get_hist_feat
from getmDWTfeat import get_mdwt_feat

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def load_data(file_path):
    logging.info(f"Loading data from {file_path}")
    data = sio.loadmat(file_path)
    emg = data['emg']
    stimulus = data['stimulus']
    repetition = data['repetition']
    return emg, stimulus, repetition

def feature_extractor(emg, stimulus, repetition, deadzone, winsize, wininc, featFunc, edges=None):
    datawin = np.ones(winsize)
    numwin = emg.shape[0]
    nSignals = emg.shape[1]

    # Allocate memory
    if featFunc == 'getHISTfeat':
        emg = (emg - np.mean(emg, axis=0)) / np.std(emg, axis=0)
        feat = np.zeros((numwin, nSignals * len(edges)), dtype=np.float32)
    elif featFunc == 'getTDfeat':
        feat = np.zeros((numwin, nSignals * 5), dtype=np.float32)
    elif featFunc == 'getmDWTfeat':
        feat = np.zeros((numwin, nSignals * 4), dtype=np.float32)
    else:
        feat = np.zeros((numwin, nSignals), dtype=np.float32)

    featStim = np.zeros((numwin, 1))
    featRep = np.zeros((numwin, 1))
    checkStimRep = np.zeros((numwin, 1))

    def process_window(winInd):
        if (winInd % wininc) == 0:
            curStimWin = stimulus[winInd:winInd + winsize]
            curRepWin = repetition[winInd:winInd + winsize]

            if np.unique(curStimWin).size == 1 and np.unique(curRepWin).size == 1:
                checkStimRep[winInd] = 1
                featStim[winInd] = curStimWin[0]
                featRep[winInd] = curRepWin[0]

                curwin = emg[winInd:winInd + winsize]

                if featFunc == 'getrmsfeat':
                    feat[winInd] = get_rms_feat(curwin, winsize, wininc)
                elif featFunc == 'getTDfeat':
                    feat[winInd] = get_td_feat(curwin, deadzone, winsize, wininc)
                elif featFunc == 'getmavfeat':
                    feat[winInd] = get_mav_feat(curwin, winsize, wininc, datawin)
                elif featFunc == 'getzcfeat':
                    feat[winInd] = get_zc_feat(curwin, deadzone, winsize, wininc, datawin)
                elif featFunc == 'getsscfeat':
                    feat[winInd] = get_ssc_feat(curwin, deadzone, winsize, wininc, datawin)
                elif featFunc == 'getwlfeat':
                    feat[winInd] = get_wl_feat(curwin, winsize, wininc, datawin)
                elif featFunc == 'getarfeat':
                    feat[winInd] = get_ar_feat(curwin, 1, winsize, wininc, datawin)
                elif featFunc == 'getiavfeat':
                    feat[winInd] = get_iav_feat(curwin, winsize, wininc, datawin)
                elif featFunc == 'getHISTfeat':
                    feat[winInd] = get_hist_feat(curwin, winsize, wininc, edges)
                elif featFunc == 'getmDWTfeat':
                    feat[winInd] = get_mdwt_feat(curwin, winsize, wininc)
                else:
                    raise ValueError("Feature function not implemented")

        return feat[winInd], featStim[winInd], featRep[winInd], checkStimRep[winInd]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_window, range(numwin - winsize)))

    for i, (f, fs, fr, c) in enumerate(results):
        feat[i], featStim[i], featRep[i], checkStimRep[i] = f, fs, fr, c

    valid_idx = checkStimRep != 0
    feat = feat[valid_idx]
    featStim = featStim[valid_idx]
    featRep = featRep[valid_idx]

    return feat, featStim, featRep

def preprocess_directory(base_dir, deadzone, winsize, wininc, featFunc):
    logging.info(f"Preprocessing directory: {base_dir}")
    for dir_name in os.listdir(base_dir):
        subject_dir = os.path.join(base_dir, dir_name)
        logging.debug(f"Checking {subject_dir}")
        if os.path.isdir(subject_dir):
            mat_files = [os.path.join(subject_dir, f) for f in os.listdir(subject_dir) if f.endswith('.mat')]
            for mat_file in mat_files:
                logging.debug(f"Processing file: {mat_file}")
                try:
                    emg, stimulus, repetition = load_data(mat_file)
                    edges = np.arange(-3, 3.3, 0.3) if featFunc == 'getHISTfeat' else None
                    feat, featStim, featRep = feature_extractor(emg, stimulus, repetition, deadzone, winsize, wininc, featFunc, edges)
                    logging.info(f"Extracted {featFunc} features for {mat_file}")
                except Exception as e:
                    logging.error(f"Error processing {mat_file}: {e}")

# Example usage
if __name__ == "__main__":
    base_directory = r'C:\Users\Hp\Desktop\EMG models\ninapro db3\s1_0\DB3_s1'
    deadzone = 1e-5
    winsize = 400
    wininc = 20
    featFunc = 'getHISTfeat'  # Example, change as needed
    preprocess_directory(base_directory, deadzone, winsize, wininc, featFunc)

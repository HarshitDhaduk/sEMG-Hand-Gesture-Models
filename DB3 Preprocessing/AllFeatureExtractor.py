import os
import numpy as np
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor
import argparse

# Feature extraction functions
def get_rms_feat(x, winsize, wininc, datawin=None, dispstatus=0):
    if datawin is None:
        datawin = np.ones(winsize)
    
    numwin = (x.shape[0] - winsize) // wininc + 1
    feat = np.zeros((numwin, x.shape[1]))

    for i in range(numwin):
        curwin = x[i * wininc: i * wininc + winsize, :] * datawin[:, None]
        feat[i, :] = np.sqrt(np.mean(curwin ** 2, axis=0))

    return feat

def get_iav_feat(x, winsize, wininc, datawin=None, dispstatus=0):
    if datawin is None:
        datawin = np.ones(winsize)
    
    numwin = (x.shape[0] - winsize) // wininc + 1
    feat = np.zeros((numwin, x.shape[1]))

    for i in range(numwin):
        curwin = x[i * wininc: i * wininc + winsize, :] * datawin[:, None]
        feat[i, :] = np.sum(np.abs(curwin), axis=0)

    return feat

def get_mav_feat(x, winsize, wininc, datawin=None, dispstatus=0):
    if datawin is None:
        datawin = np.ones(winsize)
    
    numwin = (x.shape[0] - winsize) // wininc + 1
    feat = np.zeros((numwin, x.shape[1]))

    for i in range(numwin):
        curwin = x[i * wininc: i * wininc + winsize, :] * datawin[:, None]
        feat[i, :] = np.mean(np.abs(curwin), axis=0)

    return feat

def get_wl_feat(x, winsize, wininc, datawin=None, dispstatus=0):
    if datawin is None:
        datawin = np.ones(winsize)
    
    numwin = (x.shape[0] - winsize) // wininc + 1
    feat = np.zeros((numwin, x.shape[1]))

    for i in range(numwin):
        curwin = x[i * wininc: i * wininc + winsize, :] * datawin[:, None]
        feat[i, :] = np.sum(np.abs(np.diff(curwin, axis=0)), axis=0)

    return feat

def get_ssc_feat(x, deadzone, winsize, wininc, datawin=None, dispstatus=0):
    if datawin is None:
        datawin = np.ones(winsize)
    
    x = np.vstack([np.zeros((1, x.shape[1])), np.diff(x, axis=0)])
    numwin = (x.shape[0] - winsize) // wininc + 1
    feat = np.zeros((numwin, x.shape[1]))

    for i in range(numwin):
        y = x[i * wininc: i * wininc + winsize, :] * datawin[:, None]
        y = (y > deadzone) - (y < -deadzone)
        dz = np.diff(y, axis=0)
        feat[i, :] = np.sum(np.abs(dz) == 2, axis=0)

    return feat

def get_zc_feat(x, deadzone, winsize, wininc, datawin=None, dispstatus=0):
    if datawin is None:
        datawin = np.ones(winsize)
    
    numwin = (x.shape[0] - winsize) // wininc + 1
    feat = np.zeros((numwin, x.shape[1]))

    for i in range(numwin):
        y = x[i * wininc: i * wininc + winsize, :] * datawin[:, None]
        y = (y > deadzone) - (y < -deadzone)
        dz = np.diff(y, axis=0)
        feat[i, :] = np.sum(np.abs(dz) == 2, axis=0)

    return feat

def get_td_feat(x, deadzone, winsize, wininc):
    feat1 = get_mav_feat(x, winsize, wininc)
    feat2 = get_wl_feat(x, winsize, wininc)
    feat3 = get_zc_feat(x, deadzone, winsize, wininc)
    feat4 = get_ssc_feat(x, deadzone, winsize, wininc)
    feat5 = get_iav_feat(x, winsize, wininc)
    return np.hstack([feat1, feat2, feat3, feat4, feat5])

def get_hist_feat(x, winsize, wininc, edges):
    numwin = (x.shape[0] - winsize) // wininc + 1
    feat = np.zeros((numwin, x.shape[1] * len(edges)))

    for i in range(numwin):
        curwin = x[i * wininc: i * wininc + winsize, :]
        F0 = np.array([np.histogram(curwin[:, j], bins=edges)[0] for j in range(x.shape[1])]).T
        feat[i, :] = F0.flatten()

    return feat

def get_mdwt_feat(x, winsize, wininc):
    import pywt
    numwin = (x.shape[0] - winsize) // wininc + 1
    feat = []

    for i in range(numwin):
        curwin = x[i * wininc: i * wininc + winsize, :]
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

    return np.array(feat)

# Main feature extraction function
def feature_extractor(emg, stimulus, repetition, deadzone, winsize, wininc, featFunc, edges=None):
    datawin = np.ones(winsize)
    numwin = emg.shape[0]
    nSignals = emg.shape[1]

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
        results = executor.map(process_window, range(numwin - winsize))

    for i, (f, fs, fr, c) in enumerate(results):
        feat[i], featStim[i], featRep[i], checkStimRep[i] = f, fs, fr, c

    valid_idx = checkStimRep != 0
    feat = feat[valid_idx]
    featStim = featStim[valid_idx]
    featRep = featRep[valid_idx]

    return feat, featStim, featRep

def load_data(file_path):
    print(f"Loading data from {file_path}")
    data = sio.loadmat(file_path)
    emg = data.get('emg', None)
    stimulus = data.get('stimulus', None)
    repetition = data.get('repetition', None)
    return emg, stimulus, repetition

def save_features(output_dir, subject_id, features):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'{subject_id}_features.mat')
    sio.savemat(output_file, features)
    print(f'Saved features to {output_file}')

def preprocess_directory(base_dir, output_dir, deadzone, winsize, wininc, featFunc):
    outer_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Found outer directories: {outer_dirs}")
    
    for outer_dir in outer_dirs:
        inner_dirs = [os.path.join(outer_dir, d) for d in os.listdir(outer_dir) if os.path.isdir(os.path.join(outer_dir, d))]
        print(f"Found inner directories in {outer_dir}: {inner_dirs}")
        
        for inner_dir in inner_dirs:
            subject_id = os.path.basename(inner_dir)
            all_features = {'feat': [], 'featStim': [], 'featRep': []}

            mat_files = [os.path.join(inner_dir, f) for f in os.listdir(inner_dir) if f.endswith('.mat')]
            print(f"Found .mat files in {inner_dir}: {mat_files}")
            for mat_file in mat_files:
                emg, stimulus, repetition = load_data(mat_file)
                if emg is None or stimulus is None or repetition is None:
                    print(f"Failed to load data from {mat_file}. Skipping.")
                    continue
                print(f"Loaded data from {mat_file}")
                edges = np.arange(-3, 3.3, 0.3) if featFunc == 'getHISTfeat' else None
                feat, featStim, featRep = feature_extractor(emg, stimulus, repetition, deadzone, winsize, wininc, featFunc, edges)
                
                all_features['feat'].append(feat)
                all_features['featStim'].append(featStim)
                all_features['featRep'].append(featRep)
            
            # Concatenate features from all exercises
            all_features['feat'] = np.concatenate(all_features['feat'], axis=0)
            all_features['featStim'] = np.concatenate(all_features['featStim'], axis=0)
            all_features['featRep'] = np.concatenate(all_features['featRep'], axis=0)

            save_features(output_dir, subject_id, all_features)

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract EMG features from Ninapro database.')
    parser.add_argument('--base_directory', type=str, required=True, help='Base directory of the database')
    parser.add_argument('--output_directory', type=str, required=True, help='Directory to save the extracted features')
    parser.add_argument('--deadzone', type=float, default=1e-5, help='Deadzone for feature extraction')
    parser.add_argument('--winsize', type=int, default=400, help='Window size for feature extraction')
    parser.add_argument('--wininc', type=int, default=20, help='Window increment for feature extraction')
    parser.add_argument('--featFunc', type=str, required=True, help='Feature extraction function to use')
    args = parser.parse_args()

    preprocess_directory(args.base_directory, args.output_directory, args.deadzone, args.winsize, args.wininc, args.featFunc)


# To run the script and extract features, use the following command in your terminal:
# python main.py --base_directory "C:\Users\Hp\Desktop\EMG models\ninapro db3" --output_directory "C:\Users\Hp\Desktop\EMG models\features" --featFunc "getrmsfeat"

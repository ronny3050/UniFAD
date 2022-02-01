from sklearn.metrics import roc_curve, accuracy_score
import numpy as np

def apcer(dataset_name, spoof_paths, spoof_scores, thr):
    if dataset_name == 'siw':
        return None, None, np.where(np.array(spoof_scores) < thr)[0].shape[0]/len(spoof_scores) * 100
    replay = []
    paper = []
    for i, spoof_path in enumerate(spoof_paths):
        spoof_type = int(spoof_path.split('_')[-1])
        if spoof_type == 2 or spoof_type == 3:
            paper.append(spoof_scores[i])
        else:
            replay.append(spoof_scores[i])
    replay_apcer =  np.where(replay < thr)[0].shape[0]/len(replay) * 100
    paper_apcer = np.where(paper < thr)[0].shape[0]/len(paper) * 100
    return paper_apcer, replay_apcer, max([replay_apcer, paper_apcer])

def bpcer(live_paths, live_scores, thr):
    return np.where(np.array(live_scores) >= thr)[0].shape[0]/len(live_scores) * 100.

def accuracy(live_scores, spoof_scores):
    live_labels = [0 for ls in live_scores]
    spoof_labels = [1 for ss in spoof_scores]
    y = np.array(spoof_labels + live_labels)
    scores = np.array(spoof_scores + live_scores)
    pred = np.array(scores>0.5, np.int32)
    return accuracy_score(y, pred)

def eer(live_scores, spoof_scores, fdr=0.002):
    # print('Computing ROC for {} live scores and {} spoof scores'.format(len(live_scores), len(spoof_scores)))
    live_labels = [0 for ls in live_scores]
    spoof_labels = [1 for ss in spoof_scores]
    y = np.array(spoof_labels + live_labels)
    scores = np.array(spoof_scores + live_scores)

    fpr, tpr, thresholds = roc_curve(y, scores)
    fnr = 1 - tpr
    eer_thr = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    tdr = 0.0
    for j, val in enumerate(fpr):
        if val > fdr:
            tdr = tpr[j-1]
            break
    return fpr[np.nanargmin(np.absolute((fnr-fpr)))], eer_thr, tdr

def fuse_scores(live_paths, live_scores, method='mean'):
    lives = []
    scores = []
    paths = []
    for live in live_paths:
        live = "_".join(live.split('/')[-1].split('.')[0].split('_')[:-1])
        lives.append(live)
    uq = np.unique(lives).tolist()
    for u in uq:
        idxs = [i for i, l in enumerate(live_paths) if "_".join(l.split('/')[-1].split('.')[0].split('_')[:-1]) == u]
        if method == 'sum':
            scores.append(np.average(np.array(live_scores)[idxs]))
        else:
            scores.append(np.average(np.array(live_scores)[idxs]))
        paths.append(u)
    return paths, scores

def fuse_features(live_paths, live_scores):
    lives = []
    scores = []
    paths = []
    for live in live_paths:
        live = "_".join(live.split('/')[-1].split('.')[0].split('_')[:-1])
        lives.append(live)
    uq = np.unique(lives).tolist()
    for u in uq:
        idxs = [i for i, l in enumerate(live_paths) if "_".join(l.split('/')[-1].split('.')[0].split('_')[:-1]) == u]
        scores.append(np.mean(np.array(live_scores)[idxs], axis=0, keepdims=True))
        paths.append(u)
    return paths, np.squeeze(np.array(scores))
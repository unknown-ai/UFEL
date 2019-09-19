import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset


def return_LR_weights(features_in, features_out):
    train_X = np.concatenate([features_in, features_out], axis=1).T
    train_y = np.concatenate([np.repeat(1, features_in.shape[1]),
                              np.repeat(0, features_out.shape[1])])
    cls = LogisticRegression(class_weight="balanced", C=100)
    cls.fit(train_X, train_y)
    return cls.intercept_[0], cls.coef_[0]


def return_metrics(f_x, test_loader, mask_list=None,
                   device="cuda", print_detail=False, rep=[1, 1, 1]):
    pred_y = []
    test_y = []
    softmax_arr = []
    logvar1_arr = []
    logvar2_arr = []
    logvar3_arr = []
    with torch.no_grad():
        for x, y in test_loader:
            if mask_list:
                x, y = mask(x, y, mask_list)
            softmax = F.softmax(f_x(x.to(device)), dim=1)
            if rep[0]:
                logvar1_arr.extend(f_x.logvar1.reshape(
                    softmax.size(0), -1).mean(1).detach().cpu().numpy())
            if rep[1]:
                logvar2_arr.extend(f_x.logvar2.reshape(
                    softmax.size(0), -1).mean(1).detach().cpu().numpy())
            if rep[2]:
                logvar3_arr.extend(-f_x.logvar3.reshape(
                    softmax.size(0), -1).mean(1).detach().cpu().numpy())
            max_value, argmax = softmax.max(1)
            pred_y.extend(argmax.detach().cpu().numpy())
            test_y.extend(y.numpy())
            softmax_arr.extend(max_value.detach().cpu().numpy())
    acc = accuracy_score(test_y, pred_y)
    features = [softmax_arr, logvar1_arr, logvar2_arr, logvar3_arr]
    features = [f for f in features if f]  # remove empty
    if print_detail:
        print(confusion_matrix(test_y, pred_y))
        print(classification_report(test_y, pred_y))

    if np.array(rep).sum():
        return np.stack(features), acc
    else:
        return np.array(softmax_arr), acc


def odin(f_x, dataloader, temperature, magnitude, mask_list=None, device="cuda"):
    criterion = nn.CrossEntropyLoss()
    max_softmax = []
    for x, y in dataloader:
        if mask_list:
            x, y = mask(x, y, mask_list)
        x.requires_grad = True
        logits = f_x(x.to(device))
        output = logits/temperature
        labels = output.data.max(1)[1]
        loss = criterion(output, labels)
        loss.backward()

        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        tempInputs = torch.add(x.data, -magnitude, gradient)
        output = f_x(tempInputs.to(device))
        output = output / temperature
        softmax = F.softmax(output, dim=1)
        max_softmax.extend(softmax.max(1)[0].detach().cpu().numpy())
    max_softmax = np.array(max_softmax)
    return max_softmax

def mask(x, y, mask_list):
    m = []
    for i in mask_list:
        m.append(y == i)
    m = torch.stack(m).sum(0).byte()
    x_mask = x[m]
    y_mask = y[m]
    return x_mask, y_mask


# https://github.com/pokaxpoka/deep_Mahalanobis_detector/blob/master/calculate_log.py
def get_curve(knowns, novels, stypes):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for known, novel, stype in zip(knowns, novels, stypes):
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known), np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        fp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k+num_n):
            if k == num_k:
                tp[stype][l+1:] = tp[stype][l]
                fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
                fp[stype][l+1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l+1] = tp[stype][l]
                    fp[stype][l+1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l+1] = tp[stype][l] - 1
                    fp[stype][l+1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95


def metric(knowns, novels, accs, stypes, verbose=True):
    tp, fp, tnr_at_tpr95 = get_curve(knowns, novels, stypes)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT', 'ACC']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')

    for stype in stypes:
        if verbose:
            print('{stype:5s} '.format(stype=stype), end='')
        results[stype] = dict()

        # TNR
        mtype = 'TNR'
        results[stype][mtype] = tnr_at_tpr95[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')

        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
        results[stype][mtype] = -np.trapz(1.-fpr, tpr)
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')

        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = .5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max()
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')

        # AUIN
        mtype = 'AUIN'
        denom = tp[stype]+fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
        results[stype][mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')

        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
        results[stype][mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')

        # Accuracy
        mtype = 'ACC'
        results[stype][mtype] = accs[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
            print('')

    return results


# https://github.com/uoguelph-mlrg/confidence_estimation/blob/master/utils/datasets.py
class GaussianNoise(Dataset):
    """Gaussian Noise Dataset"""

    def __init__(self, size=(3, 32, 32), n_samples=10000, mean=0.5, variance=1.0):
        self.size = size
        self.n_samples = n_samples
        self.mean = mean
        self.variance = variance
        self.data = np.random.normal(loc=self.mean, scale=self.variance, size=(self.n_samples,) + self.size)
        self.data = np.clip(self.data, 0, 1)
        self.data = self.data.astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], 0


class UniformNoise(Dataset):
    """Uniform Noise Dataset"""

    def __init__(self, size=(3, 32, 32), n_samples=10000, low=0, high=1):
        self.size = size
        self.n_samples = n_samples
        self.low = low
        self.high = high
        self.data = np.random.uniform(low=self.low, high=self.high, size=(self.n_samples,) + self.size)
        self.data = self.data.astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], 0

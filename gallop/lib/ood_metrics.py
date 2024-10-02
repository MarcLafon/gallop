import numpy as np
import sklearn.metrics as skm


def stable_cumsum(
        arr: np.ndarray,
        rtol: float = 1e-05,
        atol: float = 1e-08,
) -> np.ndarray:
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(
        y_true: np.ndarray,
        y_score: np.ndarray,
        recall_level: float = 0.95,
        pos_label: bool = None,
) -> np.ndarray:
    classes = np.unique(y_true)
    if (pos_label is None) and (not (np.array_equal(classes, [0, 1]) or np.array_equal(classes, [-1, 1]) or np.array_equal(classes, [0]) or np.array_equal(classes, [-1]) or np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_auroc(xin: np.ndarray, xood: np.ndarray) -> float:
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    return auroc


def get_det_accuracy(xin: np.ndarray, xood: np.ndarray) -> float:
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    fpr, tpr, thresholds = skm.roc_curve(labels, data)
    return .5 * (tpr + 1. - fpr).max()


def get_aupr_out(xin: np.ndarray, xood: np.ndarray) -> float:
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    aupr_out = skm.average_precision_score(labels, data)
    return aupr_out


def get_aupr_in(xin: np.ndarray, xood: np.ndarray) -> float:
    labels = [1] * len(xin) + [0] * len(xood)
    data = np.concatenate((-xin, -xood))
    aupr_in = skm.average_precision_score(labels, data)
    return aupr_in


def get_fpr(xin: np.ndarray, xood: np.ndarray, mode: str = "mcm") -> float:
    if mode == "mcm":
        examples = np.hstack((-xin, -xood))
        labels = np.zeros(len(examples), dtype=np.int32)
        labels[:len(xin)] += 1
        return fpr_and_fdr_at_recall(labels, examples)
    else:
        return np.sum(xood < np.percentile(xin, 95)) / len(xood)

import numpy as np
import numpy.lib.npyio
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
import functools
from typing import Union, Tuple, Dict

gaussian_filter1d = functools.partial(gaussian_filter1d, axis=0, mode='constant')


def cal_micro_auc(score_dict: Dict[str, np.ndarray], gt_npz: numpy.lib.npyio.NpzFile, slen: int, sitv: int, return_score_gt: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    '''
    Calculate micro-AUC.
    '''
    vid_name_list = sorted(list(score_dict.keys()))

    def _concat_score(_score_dict: dict):
        _cat_gts = []
        _cat_scores = []
        for _vid_name in vid_name_list:
            _vid_gt = gt_npz[_vid_name]
            _vid_score: np.ndarray = _score_dict[_vid_name]
            _vid_score = _vid_score.squeeze()
            assert _vid_gt.ndim == _vid_score.ndim == 1, f"{_vid_gt.shape}, {_vid_score.shape}"

            for _p in kwargs['score_post_process']:
                if _p == "filt":
                    _vid_score = gaussian_filter1d(_vid_score, sigma=sitv * slen / 2)
                elif _p == "norm":
                    _vid_score = normalize_score(_vid_score, 'minmax')
                else:
                    raise NotImplementedError(f"{_p}")

            assert len(_vid_gt) == len(_vid_score), f"{_vid_gt.shape}, {_vid_score.shape}, {slen}, {sitv}"
            _cat_gts.append(_vid_gt)
            _cat_scores.append(_vid_score)
        _cat_gts = np.concatenate(_cat_gts)
        _cat_scores = np.concatenate(_cat_scores)
        return _cat_gts, _cat_scores

    cat_gts, cat_scores = _concat_score(score_dict)

    micro_auc = roc_auc_score(cat_gts, cat_scores)

    if return_score_gt:
        return micro_auc, cat_scores, cat_gts
    return micro_auc


def normalize_score(input_score: np.ndarray, ntype: str):
    if ntype == None:
        return input_score

    assert input_score.ndim in (1, 2), f"{input_score.shape}"
    ntype = ntype.lower()

    score: np.ndarray = input_score.copy()
    if score.ndim == 1:
        score = np.expand_dims(score, 1)

    if ntype == 'minmax':
        # MinMax
        denominator = score.max(0, keepdims=True) - score.min(0, keepdims=True)
        # assert np.all(denominator != 0)
        if np.all(denominator == 0):
            print("WARNING: np.all(denominator == 0) in `normalize_score`")
            score = score
        else:
            score = (score - score.min(0, keepdims=True)) / denominator
    elif ntype == 'meanstd':
        # MeanStd
        denominator = score.std(0, keepdims=True)
        assert np.all(denominator != 0)
        score = (score - score.mean(0, keepdims=True)) / denominator
    elif ntype == 'l2norm':
        # L2Norm
        denominator = np.linalg.norm(score, ord=2, axis=0, keepdims=True)
        assert np.all(denominator != 0)
        score = score / denominator
    else:
        raise NotImplementedError(ntype)

    if input_score.ndim == 1:
        score = score.squeeze()
    assert score.shape == input_score.shape
    return score

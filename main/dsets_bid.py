from os import listdir, getpid
from os.path import join, exists, basename, dirname

import torch
from torch.utils.data.dataset import Dataset as tc_Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from typing import Tuple, Dict, List, Union
from collections import OrderedDict
import pickle
import random
import numpy as np
import mmcv
import warnings

mmcv.use_backend('turbojpeg')


class DatasetInfoDict():
    def __init__(self):
        # (h, w); None represents more than one resolutions
        self.frame_size = {'ST': (480, 856), 'Ave': (360, 640), 'Cor': (1080, 1920), 'NWPU': (1080, 1920)}

    @property
    def data_names(self):
        return ('ST', 'Ave', 'Cor', 'NWPU')

    def __getitem__(self, data_name: str):
        data_attr = {}
        for attr_name, attr_dict in self.__dict__.items():
            if not attr_name.startswith('__'):
                if data_name in attr_dict:
                    data_attr[attr_name] = attr_dict[data_name]
                else:
                    raise KeyError(f"Cannot find the attribute '{attr_name}' for '{data_name}'")
        return data_attr


class BoxTransform():
    def __init__(self, frame_height, frame_width):
        self.frame_height = frame_height
        self.frame_width = frame_width

    def __call__(self, batch_bbox: torch.Tensor):
        # batch_bbox: [C=4,T]
        assert batch_bbox.ndim == 2 and batch_bbox.shape[0] == 4, f"{batch_bbox.shape}"
        dst_batch_box = torch.zeros_like(batch_bbox)
        dst_batch_box[0] = (batch_bbox[0] + batch_bbox[2]) / 2 / self.frame_width
        dst_batch_box[1] = (batch_bbox[1] + batch_bbox[3]) / 2 / self.frame_height
        dst_batch_box[2] = (batch_bbox[2] - batch_bbox[0]) / self.frame_width
        dst_batch_box[3] = (batch_bbox[3] - batch_bbox[1]) / self.frame_height
        return dst_batch_box


class TrainSetTrackingObject(tc_Dataset):
    def __init__(self, video_dir: str, track_dir: str, snippet_len: int, snippet_itv: int, iterations: int = 1, vid_suffix='avi', cache_video: bool = False, device='cpu', **kwargs):
        super().__init__()
        self._video_dir = video_dir
        self._snippet_len = snippet_len
        self._snippet_itv = snippet_itv
        self._iterations = iterations
        self._vid_suffix = vid_suffix
        self.device = device
        self._kwargs = kwargs

        data_info_dict = DatasetInfoDict()
        self.data_info = data_info_dict[kwargs.get('data_name', list(filter(lambda dname: dname in video_dir, data_info_dict.data_names))[0])]

        self._all_trk_dict = self._get_track_dict(track_dir)
        self._vid_samp_weight = self._get_video_sampling_weight(self._all_trk_dict)
        self._frm_samp_weight = self._get_frame_sampling_weight(self._all_trk_dict)

        self._tsfm_img = self._get_tsfm_img()
        self._tsfm_box = self._get_tsfm_box()

        self._rng = random.Random(kwargs.get('seed', 2022 + getpid()))

        self._frame_dir = kwargs.get('frame_dir', "")

        self.__cache_max_vid = kwargs.get('cache_max_vid', len(self._all_trk_dict))
        self.__cache_max_frm = kwargs.get('cache_max_frm', 4096)

        if cache_video:
            warnings.warn(f"work>1时用vid_cache可能会造成内存翻倍，并产生OpenCV错误！")
            self._vid_cache = mmcv.Cache(self.__cache_max_vid)
        else:
            self._vid_cache = None

        self.check_init()

        self._frname_tmpl = self._get_frm_file_tmpl() if self._frame_dir else None

    @staticmethod
    def _get_track_dict(track_dir: str) -> OrderedDict:
        track_dict = OrderedDict()
        for pkl_name in sorted(listdir(track_dir)):
            with open(join(track_dir, pkl_name), 'rb') as f:
                _trk_dict: OrderedDict = pickle.load(f)
                track_dict[pkl_name.split('.')[0]] = _trk_dict
        return track_dict

    @staticmethod
    def _get_video_sampling_weight(all_track_dict: OrderedDict) -> List[int]:
        vid_samp_weight = [len(vid_trk_dict) for vid_trk_dict in all_track_dict.values()]
        return vid_samp_weight

    @staticmethod
    def _get_frame_sampling_weight(all_track_dict: OrderedDict) -> Dict[str, Dict[int, int]]:
        frm_samp_weight = OrderedDict()
        for vid_name, vid_trk_dict in all_track_dict.items():
            vid_trk_dict: OrderedDict
            assert not vid_name in frm_samp_weight
            frm_samp_weight[vid_name] = OrderedDict()
            for frm_idx, trk_ary in vid_trk_dict.items():
                # 之后可以用(key, value)组成的list对来确定每个frm_idx的weight
                assert not frm_idx in frm_samp_weight
                frm_samp_weight[vid_name][frm_idx] = len(trk_ary)
        return frm_samp_weight

    @property
    def snippet_len(self):
        return self._snippet_len

    @property
    def snippet_itv(self):
        return self._snippet_itv

    @property
    def vid_suffix(self):
        return self._vid_suffix

    @property
    def iterations(self):
        return self._iterations

    @property
    def vid_samp_weight(self):
        return self._vid_samp_weight

    @property
    def frm_samp_weight(self):
        return self._frm_samp_weight

    @property
    def all_trk_dict(self):
        return self._all_trk_dict

    def check_init(self):
        vid_list = sorted(listdir(self._video_dir))
        assert len(vid_list) > 0
        assert len(vid_list) == len(self.all_trk_dict) == len(self.vid_samp_weight) == len(self.frm_samp_weight)

        if self._frame_dir:
            assert exists(self._frame_dir)

        for file_name in vid_list:
            vid_name = file_name.split('.')[0]
            assert vid_name in self.all_trk_dict, f"{vid_name}"

            if self._frame_dir:
                assert exists(p := join(self._frame_dir, vid_name)), f"{p}"

                with mmcv.VideoReader(join(self._video_dir, file_name)) as v:
                    n = v.frame_cnt
                if not n == len(lp := listdir(p)):
                    print(f"WARNING: video \"{p}\": vid_len = {n}, num_frm = {len(lp)}. The difference of frame numbers is ignored. Note it might cause bugs.")

                assert np.all(np.array(alp := list(map(len, lp))) == alp[0]), f"{p}"

    @staticmethod
    def _get_tsfm_img():
        tsfm = T.Compose([
            T.Resize((256, 256)),
            T.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225], inplace=True),
        ])
        return tsfm

    def _get_tsfm_box(self):
        tsfm = BoxTransform(self.data_info['frame_size'][0], self.data_info['frame_size'][1])
        return tsfm

    def _get_frm_file_tmpl(self) -> str:
        one_frame_name = listdir(join(self._frame_dir, listdir(self._frame_dir)[0]))[0]
        frm_name, suffix = one_frame_name.split('.')
        fname_tmpl = "{:0" + str(len(frm_name)) + "d}" + ".{}".format(suffix)
        # fname_tmpl = "{:{}d}.{}".format(len(frm_name), suffix)
        return fname_tmpl

    def choose_one_video(self) -> str:
        raise NotImplementedError()
        cho_vid_name = self._rng.choices(list(self.all_trk_dict.keys()), weights=self.vid_samp_weight, k=1)[0]
        return cho_vid_name

    def choose_one_frame(self, vid_name: str, uniformly: bool = False) -> int:
        frm_idx_list = list(self.frm_samp_weight[vid_name].keys())
        if frm_idx_list == []:
            # print(vid_name, self.frm_samp_weight[vid_name])
            return None
        if uniformly:
            cho_frm_idx = self._rng.choice(frm_idx_list)
        else:
            cho_frm_idx = self._rng.choices(frm_idx_list, weights=self.frm_samp_weight[vid_name].values(), k=1)[0]
        return cho_frm_idx

    def OLD_choose_one_track(self, vid_name: str, frm_idx: int) -> List[Tuple[int, np.ndarray]]:
        trk_data: np.ndarray = self.all_trk_dict[vid_name][frm_idx]
        trk_ary = trk_data[self._rng.randrange(0, len(trk_data))]  # trk_id(0), bbox(1~4), prob(5), cls(6)
        trk_id = trk_ary[0]

        snippet_trk_ary_list = [(frm_idx, trk_ary)]

        for _fi in [frm_idx - i * self.snippet_itv for i in range(1, self.snippet_len)]:
            _fi = round(_fi)
            if _fi in self.all_trk_dict[vid_name]:
                _other_trk_data = self.all_trk_dict[vid_name][_fi]
                _ret_idx = np.where(trk_id == _other_trk_data[:, 0])
                assert len(_ret_idx) == 1, f"{len(_ret_idx)}"
                if len(_ret_idx[0]) == 0:
                    return []
                else:
                    _ary_idx = _ret_idx[0][0]
                    _other_trk_ary = self.all_trk_dict[vid_name][_fi][_ary_idx]
                    snippet_trk_ary_list.append((_fi, _other_trk_ary))
            else:
                return []

        snippet_trk_ary_list.reverse()

        return snippet_trk_ary_list

    def choose_one_track(self, vid_name: str, frm_idx: int) -> List[Tuple[int, np.ndarray]]:
        trk_data: np.ndarray = self.all_trk_dict[vid_name][frm_idx]
        trk_ary = trk_data[self._rng.randrange(0, len(trk_data))]  # trk_id(0), bbox(1~4), prob(5), cls(6)
        trk_id = trk_ary[0]

        snippet_trk_ary_list = [(frm_idx, trk_ary)]

        for _fi in [frm_idx - i * self.snippet_itv for i in range(1, self.snippet_len)]:
            _fi = round(_fi)
            if _fi in self.all_trk_dict[vid_name]:
                _other_trk_data = self.all_trk_dict[vid_name][_fi]
                _ret_idx = np.where(trk_id == _other_trk_data[:, 0])
                assert len(_ret_idx) == 1, f"{len(_ret_idx)}"
                if len(_ret_idx[0]) == 0:
                    return []
                else:
                    _ary_idx = _ret_idx[0][0]
                    _other_trk_ary = self.all_trk_dict[vid_name][_fi][_ary_idx]
                    snippet_trk_ary_list.append((_fi, _other_trk_ary))
            else:
                return []

        snippet_trk_ary_list.reverse()

        return snippet_trk_ary_list

    def sample_one_snippet_track(self, vid_name: str) -> List[Tuple[int, np.ndarray]]:
        # vid_name = self.choose_one_video()
        # end_frm_idx = self.choose_one_frame(vid_name)
        if (end_frm_idx := self.choose_one_frame(vid_name)) is None:
            # print(f"{vid_name}: EMPTY!")
            raise TimeoutError(f"{vid_name}")
        snippet_trk_ary_list = self.choose_one_track(vid_name, end_frm_idx)
        n_attempt = 30
        while snippet_trk_ary_list == []:
            n_attempt -= 1
            if n_attempt == 0:
                raise TimeoutError(f"{vid_name}")
                # raise TimeoutError(f"{vid_name}")
            end_frm_idx = self.choose_one_frame(vid_name, n_attempt <= 5)
            snippet_trk_ary_list = self.choose_one_track(vid_name, end_frm_idx)
        return snippet_trk_ary_list

    def _read_from_image(self, video_path: str, frame_idx: int) -> np.ndarray:
        frm_path = join(self._frame_dir, basename(video_path).split('.')[0], self._frname_tmpl.format(frame_idx))
        assert exists(frm_path), f"{frm_path}"
        return mmcv.imread(frm_path, backend='turbojpeg')

    def load_snippet(self, vid_name: str, snippet_trk_ary_list: List[Tuple[int, np.ndarray]]) -> torch.Tensor:
        vid_path = join(self._video_dir, f"{vid_name}.{self.vid_suffix}")
        if not exists(vid_path):
            raise FileNotFoundError(vid_path)

        vid_cap = None
        if self._frame_dir:
            pass
        else:
            if self._vid_cache:
                if _v := self._vid_cache.get(vid_name):
                    vid_cap = _v
                else:
                    vid_cap = mmcv.VideoReader(vid_path, cache_capacity=self.__cache_max_frm)
                    self._vid_cache.put(vid_name, vid_cap)
            else:
                vid_cap = mmcv.VideoReader(vid_path, cache_capacity=self.__cache_max_frm)

        # 根据某一帧，确定一个方形固定视野
        obj_bbox_list = np.asarray([[max(int(p), 0) for p in _trk_ary[1:5]] for _fi, _trk_ary in snippet_trk_ary_list], np.int64)
        # 这里先根据中间帧设置
        anchor_box = obj_bbox_list[int(len(obj_bbox_list) / 2), :]

        view_center_xy = [(anchor_box[0] + anchor_box[2]) // 2, (anchor_box[1] + anchor_box[3]) // 2]

        # full_view_range = 256
        full_view_range = 512
        half_view_range = full_view_range // 2

        _fidx = snippet_trk_ary_list[0][0]
        fh, fw, _ = self._read_from_image(vid_path, _fidx).shape if self._frame_dir else vid_cap.get_frame(_fidx).shape

        view_center_xy[0] = min(max(half_view_range, view_center_xy[0]), fw - half_view_range)
        view_center_xy[1] = min(max(half_view_range, view_center_xy[1]), fh - half_view_range)

        full_view_box = [view_center_xy[0] - half_view_range, view_center_xy[1] - half_view_range,
                         view_center_xy[0] + half_view_range, view_center_xy[1] + half_view_range]

        # 然后统一读取这一个区域
        snippet = []
        for _fi, _trk_ary in snippet_trk_ary_list:
            frm: np.ndarray = self._read_from_image(vid_path, _fi) if self._frame_dir else vid_cap.get_frame(_fi)
            obj: np.ndarray = frm[full_view_box[1]:full_view_box[3], full_view_box[0]:full_view_box[2], :]

            # mmcv.imwrite(obj, f"../save.visual/main3a_{vid_name}/src_{_fi}_{_trk_ary[0]}.jpg", auto_mkdir=True)

            obj: torch.Tensor = torch.as_tensor(obj, dtype=torch.float32, device=self.device)
            obj = self._tsfm_img(obj.permute(2, 0, 1).div(255.))

            # mmcv.imwrite(np.transpose(np.array((obj.cpu().numpy() * 0.225 + 0.45) * 255, dtype=np.uint8), (1, 2, 0)), f"../save.visual/{vid_name}/restore_{_fi}_{_trk_ary[0]}.jpg")

            snippet.append(obj)

        return torch.stack(snippet, 1)

    def __len__(self) -> int:
        return len(self.all_trk_dict) * self.iterations

    def __getitem__(self, vid_idx: int):
        vid_name = list(self.all_trk_dict.keys())[vid_idx % len(self.all_trk_dict)]
        nn = 0
        while True:
            nn += 1
            try:
                snippet_trk_ary_list = self.sample_one_snippet_track(vid_name)
            except TimeoutError as e:
                # print(f"{e} Try another video.")
                vid_name = list(self.all_trk_dict.keys())[random.randrange(0, len(self.all_trk_dict))]
            else:
                break
        # if nn > 1:
        #     print(f"attempt: {vid_name}: {nn}")
        # bbox = torch.stack([torch.from_numpy(_trk_ary[1:5]) for _fi, _trk_ary in snippet_trk_ary_list], 1).float()  # [C=4, T]
        # bbox = self._tsfm_box(bbox)

        snippet = self.load_snippet(vid_name, snippet_trk_ary_list)  # [C, T, H, W]
        # print("GET_ITEM", bbox.shape, snippet.shape)
        # assert bbox.shape[1] == snippet.shape[1]

        # return (snippet, bbox)
        return snippet
        # return 1


class SnippetVideoReader(mmcv.VideoReader):
    def __init__(self, video_path: str, video_track: OrderedDict, snippet_len: int, snippet_itv: int, device='cpu', tsfm_img=None, tsfm_box=None, frame_dir: str = '', frname_tmpl: str = ''):
        super().__init__(video_path, (snippet_len + 1) * snippet_itv)
        self._vid_name = basename(video_path).split('.')[0]
        self._vid_trk = video_track
        self._slen = snippet_len
        self._sitv = snippet_itv
        self.device = device

        self._tsfm_img = tsfm_img
        self._tsfm_box = tsfm_box

        self._frame_dir = frame_dir
        self._frname_tmpl = frname_tmpl

        if self.frame_cnt < 3050:
            self.all_frames = self.read_all_frames()
            if len(self.all_frames) < 2500:
                try:
                    self.all_frames = self.all_frames.to(device)
                except RuntimeError as e:
                    print(f"{self.vid_name}: {len(self.all_frames)} frames to 'cpu'")
                    torch.cuda.empty_cache()
        else:
            self.all_frames = None

    @property
    def vid_name(self):
        return self._vid_name

    @property
    def vid_trk(self):
        return self._vid_trk

    @property
    def snippet_len(self):
        return self._slen

    @property
    def snippet_itv(self):
        return self._sitv

    def get_all_tracks(self, frm_idx: int) -> List[Tuple[int, np.ndarray]]:
        if not frm_idx in self.vid_trk:
            return []

        all_snippet_trks = []

        trk_data: np.ndarray = self.vid_trk[frm_idx]
        for ary_idx in range(0, len(trk_data)):
            trk_ary = trk_data[ary_idx]  # trk_id(0), bbox(1~4), prob(5), cls(6)
            trk_id = trk_ary[0]

            snippet_trk_ary_list = [(frm_idx, trk_ary)]
            for _fi in [frm_idx - i * self.snippet_itv for i in range(1, self.snippet_len)]:
                _fi = round(_fi)
                if _fi in self.vid_trk:
                    _other_trk_data = self.vid_trk[_fi]
                    _ret_idx = np.where(trk_id == _other_trk_data[:, 0])
                    assert len(_ret_idx) == 1, f"{len(_ret_idx)}"
                    if len(_ret_idx[0]) == 0:
                        break
                    else:
                        _ary_idx = _ret_idx[0][0]
                        _other_trk_ary = self.vid_trk[_fi][_ary_idx]
                        snippet_trk_ary_list.append((_fi, _other_trk_ary))
                else:
                    break

            if len(snippet_trk_ary_list) < self.snippet_len:
                continue

            snippet_trk_ary_list.reverse()
            all_snippet_trks.append(snippet_trk_ary_list)

        return all_snippet_trks

    def _read_from_image(self, frame_idx: int) -> np.ndarray:
        if (frm := self._cache.get(frame_idx)) is not None:
            # print("USE CACHE")
            return frm
        else:
            # print("READ")
            frm_path = join(self._frame_dir, self.vid_name, self._frname_tmpl.format(frame_idx))
            assert exists(frm_path), f"{frm_path}"
            frm = mmcv.imread(frm_path, backend='turbojpeg')
            self._cache.put(frame_idx, frm)
            return frm

    def read_all_frames(self):
        all_frames = []
        for frm_idx in range(self.frame_cnt):
            all_frames.append(torch.from_numpy(self.get_frame(frm_idx)))
        all_frames = torch.stack(all_frames)
        return all_frames  # [T, H, W, C]

    def load_snippet(self, snippet_trk_ary_list: List[Tuple[int, np.ndarray]]) -> torch.Tensor:
        # 根据某一帧，确定一个方形固定视野
        obj_bbox_list = np.asarray([[max(int(p), 0) for p in _trk_ary[1:5]] for _fi, _trk_ary in snippet_trk_ary_list], np.int64)
        # 这里先根据中间帧设置
        anchor_box = obj_bbox_list[int(len(obj_bbox_list) / 2), :]
        view_center_xy = [(anchor_box[0] + anchor_box[2]) // 2, (anchor_box[1] + anchor_box[3]) // 2]

        # full_view_range = 256
        full_view_range = 512
        half_view_range = full_view_range // 2

        _fidx = snippet_trk_ary_list[0][0]
        fh, fw, _ = self._read_from_image(_fidx).shape if self._frame_dir else self.get_frame(_fidx).shape
        view_center_xy[0] = min(max(half_view_range, view_center_xy[0]), fw - half_view_range)
        view_center_xy[1] = min(max(half_view_range, view_center_xy[1]), fh - half_view_range)

        full_view_box = [view_center_xy[0] - half_view_range, view_center_xy[1] - half_view_range,
                         view_center_xy[0] + half_view_range, view_center_xy[1] + half_view_range]

        # 然后统一读取这一个区域
        snippet = []
        for _fi, _trk_ary in snippet_trk_ary_list:
            if self.all_frames is None:
                frm: np.ndarray = self._read_from_image(_fi) if self._frame_dir else self.get_frame(_fi)
            else:
                frm: torch.Tensor = self.all_frames[_fi]

            obj = frm[full_view_box[1]:full_view_box[3], full_view_box[0]:full_view_box[2], :]

            # mmcv.imwrite(obj, f"../save.visual/{self.vid_name}/src_{_trk_ary[0]}_{_fi}.jpg", auto_mkdir=True)

            obj: torch.Tensor = torch.as_tensor(obj, dtype=torch.float32, device=self.device)
            obj = self._tsfm_img(obj.permute(2, 0, 1).div(255.))

            # mmcv.imwrite(np.transpose(np.array((obj.cpu().numpy() * 0.225 + 0.45) * 255, dtype=np.uint8), (1, 2, 0)), f"../save.visual/{self.vid_name}/restore_{_trk_ary[0]}_{_fi}.jpg")

            snippet.append(obj)

        return torch.stack(snippet, 1)  # [C,T,H,W]

    def __getitem__(self, end_frm_idx: int) -> torch.Tensor:
        '''
        把这个片段中所有物体的片段作为一个batch返回
        '''
        assert 0 <= end_frm_idx < self.frame_cnt
        if end_frm_idx < (self._slen - 1) * self._sitv:
            return None
        else:
            all_trk_list = self.get_all_tracks(end_frm_idx)
            if all_trk_list:
                batch_snippet = []
                # batch_box = []
                for _trk_list in all_trk_list:
                    batch_snippet.append(self.load_snippet(_trk_list))
                    # batch_box.append(self._tsfm_box(torch.stack([torch.from_numpy(_trk_ary[1:5]).float() for _fi, _trk_ary in _trk_list], 1)))  # [C=4,T]
                batch_snippet = torch.stack(batch_snippet, 0)  # [B,C,T=1,H,W]
                # batch_box = torch.stack(batch_box, 0)  # [B,C=4,T]
                # print("TEST-GET_ITEM", batch_snippet.shape, batch_box.shape)
                # return batch_snippet, batch_box
                return batch_snippet
            else:
                return None

    def __iter__(self):
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()


class TestSetTrackingObject(TrainSetTrackingObject):
    def __init__(self, video_dir: str, track_dir: str, snippet_len: int, snippet_itv: int, vid_suffix='avi', device='cpu', **kwargs):
        super().__init__(video_dir, track_dir, snippet_len, snippet_itv, 1, vid_suffix, False, device, **kwargs)

    def __getitem__(self, vid_idx: int):
        vid_name = list(self.all_trk_dict.keys())[vid_idx]
        vid_stream = SnippetVideoReader(join(self._video_dir, f"{vid_name}.{self.vid_suffix}"), self.all_trk_dict[vid_name],
                                        self.snippet_len, self.snippet_itv, self.device, self._tsfm_img, self._tsfm_box, self._frame_dir, self._frname_tmpl)
        return vid_stream


if __name__ == "__main__":
    seed = 2
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    from torch.utils.data import DataLoader, WeightedRandomSampler

    trainset = TrainSetTrackingObject("../data.Ave_hd/videos/Train", "../data.Ave_hd/tracking/Train", 9, 2, iterations=60, cache_video=False, device='cpu', frame_dir="../data.Ave_hd/frames/Train")

    smpl_weight = trainset.vid_samp_weight * trainset.iterations
    assert len(smpl_weight) == len(trainset)
    weighted_sampler = WeightedRandomSampler(smpl_weight, len(trainset), replacement=True)
    dloader = DataLoader(trainset, sampler=weighted_sampler, batch_size=1, pin_memory=True, num_workers=16)

    for i, (inp_img, tgt_img, inp_box, tgt_box) in enumerate(dloader):
        inp_img: torch.Tensor
        tgt_img: torch.Tensor
        inp_box: torch.Tensor
        tgt_box: torch.Tensor
        inp_img = inp_img.cuda(non_blocking=True)
        tgt_img = tgt_img.cuda(non_blocking=True)
        print(inp_img.shape, tgt_img.shape, inp_img.min(), inp_img.max(), inp_img.mean(), inp_img.std())
        print(inp_box.shape, tgt_box.shape, inp_box.min(), inp_box.max(), inp_box.mean(), inp_box.std())
        break

    testset = TestSetTrackingObject("../data.Ave_mem/videos/Test", "../data.Ave_mem/tracking/Test", 9, 2, device='cpu', frame_dir="../data.Ave_mem/frames/Test")
    # testset = TestSetTrackingObject("../data.Ave_mem/videos/Test", "../data.Ave_mem/tracking/Test", 9, 2, device='cpu')
    import time
    from time import time as ttime
    for vid_stream in testset:
        print(vid_stream.vid_name)
        vid_stream.device = 'cuda:0'
        t0 = ttime()
        for frm_idx in range(len(vid_stream)):
            batch_snippet, batch_box = vid_stream[frm_idx]
            if not batch_snippet is None:
                print(f"{frm_idx}, {batch_snippet.shape}, {batch_snippet.dtype}, {batch_snippet.device}, {batch_snippet.min()}, {batch_snippet.max()}, {batch_snippet.mean()}")
                print(f"{frm_idx}, {batch_box.shape}, {batch_box.dtype}, {batch_box.device}, {batch_box.min()}, {batch_box.max()}, {batch_box.mean()}")
            else:
                print(f"{frm_idx}, None")
        print(ttime() - t0)
        break

from torch.utils.data import Dataset
import json
import os
from utils.feature_utils import CQTconfig
from collections import defaultdict
import numpy as np
import torch


class F0TrackingDataset(Dataset):
    FEATURE_LEN = 50

    def __init__(self, root: str = './dataset/features', feature_hop: int = 50,
                 len_song: int = None):
        self.root = root
        self.feature_hop = feature_hop

        self.metadata = self._load_metadata()
        self.cqt_config = self._load_cqt_config()
        self.dataset_meta = self.metadata['dataset']
        self.id_to_filename_map = dict()
        self.by_song_id = self._load_dataset(len_song)
        self.by_idx = self._construct_idx()

    def _load_metadata(self):
        with open(os.path.join(self.root, 'metadata.json'), 'r') as f:
            return json.load(f)

    def _load_cqt_config(self):
        cqt_config = self.metadata['cqt_config']
        return CQTconfig(sr=cqt_config['sr'],
                         hop_length=cqt_config['hop_length'],
                         fmin_note=cqt_config['fmin_note'],
                         n_bins=cqt_config['n_bins'],
                         bins_per_octave=cqt_config['bins_per_octave'])

    def _load_dataset(self, len_song: int = -1):
        by_song_id = defaultdict(dict)
        for song_id, (filename, save_meta) in enumerate(self.dataset_meta.items()):
            self.id_to_filename_map[song_id] = filename

            hcqt_filepath = save_meta['hcqt_path']
            f0_filepath = save_meta['f0_path']
            hcqt = np.abs(np.load(hcqt_filepath))
            f0 = np.load(f0_filepath)

            by_song_id[song_id]['hcqt'] = hcqt
            by_song_id[song_id]['f0'] = f0

            assert hcqt.shape[2] == f0.shape[1]

            if song_id + 1 == len_song:
                break


        return by_song_id

    def _calc_num_features(self, num_frames: int):
        return (num_frames - self.FEATURE_LEN) // self.feature_hop + 1

    def _construct_idx(self):
        by_idx = defaultdict(dict)
        idx = 0
        for song_id in self.by_song_id.keys():
            hcqt = self.by_song_id[song_id]['hcqt']
            f0 = self.by_song_id[song_id]['f0']
            num_frames = hcqt.shape[2]
            num_features = self._calc_num_features(num_frames)

            for i in range(num_features):
                start_frame = i * self.feature_hop
                end_frame = start_frame + self.FEATURE_LEN
                by_idx[idx]['hcqt'] = hcqt[:, :, start_frame: end_frame]
                by_idx[idx]['f0'] = f0[:, start_frame: end_frame]
                idx += 1

        return by_idx

    def __getitem__(self, idx):
        return self.by_idx[idx]


class ToTensor(object):
    def __call__(self, sample):
        hcqt = sample['hcqt']
        f0 = sample['f0']

        return {'hcqt': torch.from_numpy(hcqt[np.newaxis, ...]),
                'f0': torch.from_numpy(f0[np.newaxis, ...])}


if __name__ == '__main__':
    dataset = F0TrackingDataset(len_song=10)
    print(dataset[10]['hcqt'].shape, dataset[10]['f0'].shape)

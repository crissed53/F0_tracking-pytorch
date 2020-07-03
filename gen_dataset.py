import csv
import json
import os
from typing import Tuple

import librosa
import numpy as np
import tqdm

from utils.feature_utils import gen_hcqt, CQTconfig, hz_to_cqt_bin

# from threading import Thread
# from queue import Queue
from multiprocessing import Process, Queue


def save_data(hcqt: np.ndarray, f0_arr: np.ndarray,
              file_meta: dict, save_root: str) -> dict:
    dirname = f'{file_meta["artist"]}_{file_meta["title"]}'
    if not os.path.exists(os.path.join(save_root, dirname)):
        os.makedirs(os.path.join(save_root, dirname))

    hcqt_save_path = save_hcqt(hcqt, dirname)
    f0_save_path = save_f0(f0_arr, dirname)

    return {'hcqt_path': hcqt_save_path,
            'f0_path': f0_save_path}


def save_hcqt(hcqt: np.ndarray, dirname: str) -> str:
    filename = f'{dirname}_hcqt'
    save_path = os.path.join(save_root, dirname, filename)
    np.save(save_path, hcqt)
    return save_path + '.npy'


def save_f0(f0_arr: np.ndarray, dirname: str) -> str:
    filename = f'{dirname}_f0'
    save_path = os.path.join(save_root, dirname, filename)
    np.save(save_path, f0_arr)
    return save_path + '.npy'


def create_dataset(root: str, metadata: dict,
                   h_factor: Tuple[float] = (0.5, 1, 2, 3),
                   cqt_config: CQTconfig = CQTconfig(),
                   save_root: str = './dataset/features') -> None:
    meta_data = {'dataset': {}, 'cqt_config': cqt_config.to_dict()}
    prog_bar = tqdm.tqdm(
        total=len(metadata), desc='creating dataset', position=0)

    for filename, file_meta in metadata.items():
        hcqt = wav_to_hcqt(root, file_meta, h_factor, cqt_config)
        f0_oh = csv_to_arr_f0(root, file_meta, cqt_config)
        frame_diff = check_if_valid_frames(hcqt, f0_oh, filename)
        if frame_diff > 0:
            hcqt = hcqt[:, :, :-frame_diff]
        elif frame_diff < 0:
            f0_oh = f0_oh[:, :frame_diff]
        save_meta = save_data(hcqt, f0_oh, file_meta, save_root)
        meta_data['dataset'][filename] = save_meta
        prog_bar.update()

    dataset_metadata_path = os.path.join(save_root, 'metadata.json')
    with open(dataset_metadata_path, 'w') as f:
        json.dump(meta_data, f, indent=4)


class DatasetCreator(Process):
    def __init__(self, in_q: Queue, out_q: Queue, prog_bar: tqdm.tqdm):
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.prog_bar = prog_bar

    def run(self):
        while True:
            param = self.in_q.get()
            if param is None:
                return
            root, file_meta, filename,  h_factor, cqt_config = param

            hcqt = wav_to_hcqt(root, file_meta, h_factor, cqt_config)
            f0_oh = csv_to_arr_f0(root, file_meta, cqt_config)

            frame_diff = check_if_valid_frames(hcqt, f0_oh, filename)
            if frame_diff > 0:
                hcqt = hcqt[:, :, :-frame_diff]
            elif frame_diff < 0:
                f0_oh = f0_oh[:, :frame_diff]
            save_meta = save_data(hcqt, f0_oh, file_meta, save_root)

            self.out_q.put((filename, save_meta))

            self.prog_bar.update()


def create_dataset_mt(root: str, metadata: dict,
                      h_factor: Tuple[float] = (0.5, 1, 2, 3),
                      cqt_config: CQTconfig = CQTconfig(),
                      save_root: str = './dataset/features',
                      num_workers: int = 32) -> None:
    in_q = Queue()
    out_q = Queue()

    prog_bar = tqdm.tqdm(
        total=len(metadata), desc='creating dataset', position=0)

    for idx, (filename, file_meta) in enumerate(metadata.items()):
        in_q.put_nowait((root, file_meta, filename, h_factor, cqt_config))

    for _ in range(num_workers):
        in_q.put(None)

    workers = [DatasetCreator(in_q, out_q, prog_bar) for _ in range(num_workers)]

    for w in workers:
        w.start()

    while not in_q.empty():
        continue

    for w in workers:
        w.join()

    meta_data = {'dataset': {}, 'cqt_config': cqt_config.to_dict()}
    while not out_q.empty():
        filename, save_meta = out_q.get()
        meta_data['dataset'][filename] = save_meta

    dataset_metadata_path = os.path.join(save_root, 'metadata.json')
    with open(dataset_metadata_path, 'w') as f:
        json.dump(meta_data, f, indent=4)


def check_if_valid_frames(hcqt: np.ndarray, f0_oh: np.ndarray, filename: str) -> int:
    """
    Check if numbers of frames in generated hcqt and one-hot vector representation
    of f0
    Args:
        hcqt:
        f0_oh:
        filename:

    Returns:

    """
    # from (h, f, t)
    hcqt_t_frame = hcqt.shape[2]
    # from (f0, t)
    f0_t_frame = f0_oh.shape[1]

    frame_diff = abs(hcqt_t_frame - f0_t_frame)

    if frame_diff > 5:
        raise ValueError(f'Too much frame diff btw hctq and f0, {frame_diff}, '
                         f'indicating corruption of data: {filename}')

    return hcqt_t_frame - f0_t_frame


def csv_to_arr_f0(root: str, file_meta: dict,
                  cqt_config: CQTconfig = CQTconfig()) -> np.ndarray:
    """
    Converts f0 information from csv to one-hot vector in np.ndarray file format
    Args:
        root: root directory of dataset
        file_meta: metadata of a single audio data

    Returns:
        f0 info in np.ndarray

    """
    melody_relpath = file_meta['melody1_path']
    melody_filepath = os.path.join(root, melody_relpath)
    with open(melody_filepath, newline='\n') as f:
        reader = csv.reader(f, delimiter=',')
        reader = list(reader)
    _, f0_list = list(zip(*reader))
    # Downsample by factor of 2: time resolution of hcqt is 11.6 ms while
    # time resolution of f0 data is half of that, 5.8 ms.
    f0_list = [float(f0) for f0 in f0_list][::2]
    # initialize one-hot vector representation of f0
    # add one more bin in frequency dim in order to remove 0 component of f0
    oh_f0 = np.zeros((cqt_config.n_bins + 1, len(f0_list)))

    f0_bins = np.array([hz_to_cqt_bin(f0, cqt_config.fmin_note,
                                      cqt_config.bins_per_octave,
                                      cqt_config.n_bins)
                        for f0 in f0_list])
    # do one-hot encoding for each f0 bin
    oh_f0[f0_bins, np.arange(f0_bins.size)] = 1

    # remove last dimension to remove zero component of f0
    return oh_f0[:-1]


def wav_to_hcqt(root: str, file_meta: dict,
                h_factor: Tuple[float] = (0.5, 1, 2, 3),
                cqt_config: CQTconfig = CQTconfig()) -> np.ndarray:
    """
    Convert wav file into HCQT given metadata for the audio dataset.
    Args:
        root: root directory in which wav and meta files are saved
        file_meta: metadata of a single audio data
        h_factor: harmonic factors in which HCQT is to be computed
        cqt_config: config to generate HCQT

    Returns:
        hcqt array

    """

    y, sr = librosa.load(os.path.join(root, file_meta['audio_path']),
                         sr=cqt_config.sr)

    hcqt = gen_hcqt(y, h_factor, cqt_config)

    return hcqt


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('-r', '--root', default='./dataset')
    p.add_argument('-d', '--dbname', default='MedleyDB-Melody')
    p.add_argument('-m', '--meta_filename',
                   default='medleydb_melody_metadata.json')
    p.add_argument('-s', '--save_root', default='./dataset/features')

    args = p.parse_args()
    root: str = args.root
    dbname: str = args.dbname
    meta_filename: str = args.meta_filename
    save_root: str = args.save_root

    meta_file = os.path.join(root, dbname, meta_filename)

    with open(meta_file) as f:
        metadata = json.load(f)

    create_dataset_mt(root, metadata, save_root=save_root)

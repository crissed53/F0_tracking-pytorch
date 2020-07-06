import collections
import os
from dataclasses import dataclass
from typing import Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class CQTconfig:
    sr: int = 22050
    hop_length: int = 256
    fmin_note: str = 'C1'
    n_bins: int = 360
    bins_per_octave: int = 60

    def to_dict(self):
        return {'sr': self.sr,
                'hop_length': self.hop_length,
                'fmin_note': self.fmin_note,
                'n_bins': self.n_bins,
                'bins_per_octave': self.bins_per_octave}


@dataclass
class AudioFileInfo:
    audio_path: str
    start_t: float
    end_t: float


def gen_hcqt(y: np.ndarray, h_factor: Tuple[float] = (0.5, 1, 2, 3),
             cqt_config: CQTconfig = CQTconfig()) -> np.ndarray:
    """
    Computes Harmonic CQT. The implementation does not directly follow the paper
    as higher up the harmonic channel, upper bound of frequency gets too large
    for given configuration. For this, h_factor = [0.5, 1, 2, 3] is recommended
    instead of [0.5, 1, 2, 3, 4, 5]
    Args:
        y: mono-audio array
        h_factor: harmonic factor which is used to compute HCQT
        cqt_config: config needed for computing CQT

    Returns:

    """
    stack = []
    min_frame = 1e+10
    for h in h_factor:
        fmin = h * librosa.note_to_hz(cqt_config.fmin_note)
        cqt = librosa.cqt(y, sr=cqt_config.sr,
                          hop_length=cqt_config.hop_length,
                          fmin=fmin, n_bins=cqt_config.n_bins,
                          bins_per_octave=cqt_config.bins_per_octave)

        if cqt.shape[1] < min_frame:
            min_frame = cqt.shape[1]

        stack.append(cqt)

    # For larger harmonic h, there is slight chance that number of frame is
    # diminished due to enlarged window length used for cqt. To compensate for
    # this, trim the frame appropriately
    stack = [cqt[:, :min_frame] for cqt in stack]
    return np.stack(stack, axis=0)


def cqt_bin_to_hz(bin: int, fmin_note: str, bins_per_octave: int):
    """
    Convert cqt bin into frequency in hz
    Args:
        bin:
        fmin_note:
        bins_per_octave:

    Returns:

    """
    bins_per_semitone = bins_per_octave / 12
    midi_number = bin / bins_per_semitone
    midi_number += librosa.note_to_midi(fmin_note)
    return librosa.midi_to_hz(midi_number)


def cqt_bin_to_midi_hz(bin: int, fmin_note: str, bins_per_octave: int):
    """
    Convert cqt bin into frequency in hz corresponding to the nearest integer
    midi number
    Args:
        bin:
        fmin_note:
        bins_per_octave:

    Returns:

    """
    midi_num = cqt_bin_to_int_midi(bin, fmin_note, bins_per_octave)
    return librosa.midi_to_hz(midi_num)


def cqt_bin_to_int_midi(bin: int, fmin_note: str, bins_per_octave: int) -> int:
    """
    Convert cqt bin into midi number rounded to the nearest integer
    Args:
        bin:
        fmin_note:
        bins_per_octave:

    Returns:

    """
    bins_per_semitone = bins_per_octave / 12
    midi_number = bin / bins_per_semitone
    midi_number += librosa.note_to_midi(fmin_note)
    return int(midi_number) if midi_number % 1 < 0.5 else int(midi_number) + 1


def f0_contour_to_midi_contour(f0_contour: np.ndarray) -> np.ndarray:
    """
    Convert f0 contour into midi (rounded to nearest integer) contour
    Args:
        f0_contour: contour of f0, in array

    Returns:

    """
    cqt_config = CQTconfig()
    bin_num = np.argmax(f0_contour, axis=0)
    bin_num = [val if val != 0 else -1 for val in bin_num]
    stack = []
    midis = []
    for bin in bin_num:
        if bin == -1:
            audio_chunk = np.array([0 for _ in range(cqt_config.hop_length)])
            stack.append(audio_chunk)
            midis.append(-1)
        else:
            midi_int = cqt_bin_to_int_midi(
                bin, cqt_config.fmin_note, cqt_config.bins_per_octave)
            midis.append(midi_int)

    midis = np.array(midis)
    return midis


def hz_to_cqt_bin(f: float, fmin_note: str, bins_per_octave: int, n_bins: int):
    """
    Converts frequency in hz to bin number of cqt
    Args:
        f: frequency to convert from
        fmin_note: midi note number for the mininum frequency
            in which cqt is computed
        bins_per_octave: bins per octave
    Returns:

    """
    if f == 0.0:
        return -1
    bins_per_semitone = bins_per_octave / 12
    # midi number of zero-th bin, eg) 'C1' -> 24
    midi_offset = int(librosa.note_to_midi(fmin_note))
    # midi number of frequency of interest, eg) 446 Hz -> 69.23
    f_midi = librosa.hz_to_midi(f)

    if f_midi < midi_offset:
        raise ValueError(f'f should be larger than frequency corresponding '
                         f'to fmin_note. Got f: {f} and fmin_note: '
                         f'{librosa.note_to_hz(fmin_note)}')
    f_midi_int = int(f_midi // 1)
    # decimal point of the midi number, eg) 69.23 -> 0.23
    f_midi_frac = f_midi % 1
    # corresponding offset, in bin, from the bin corresponding to the  integer
    # of the midi number, eg) 0.23 -> 1
    rel_bin = f_midi_frac * bins_per_semitone
    if rel_bin % 1 > 0.5:
        rel_bin = int(rel_bin) + 1
    else:
        rel_bin = int(rel_bin)

    bin_num = int((f_midi_int - midi_offset) * bins_per_semitone) + rel_bin
    # if bin number is larger than the upper bound, return -1 so that it will
    # not be one-hot encoded
    if bin_num + 1 > n_bins:
        return -1
    else:
        return bin_num


def visualize_f0_and_cqt(
        f0_arr: np.ndarray, cqt: np.ndarray, afile_info: AudioFileInfo) -> None:
    """
    Visualize both f0 contour and cqt
    Args:
        f0_arr: f0 contour in array
        cqt: cqt
        afile_info: information of audio file

    Returns:

    """
    filename = os.path.basename(afile_info.audio_path)
    plt.figure(figsize=(20, 10))
    plt.subplot(311)
    librosa.display.specshow(
        f0_arr, x_axis='s', y_axis='cqt_note', sr=22050, hop_length=256,
        fmin=librosa.note_to_hz('C1'), bins_per_octave=60)
    plt.title(f'Extracted f0: {filename.strip(".wav")} '
              f'from {afile_info.start_t}s to {afile_info.end_t}s')
    plt.set_cmap('binary')

    plt.subplot(312)
    librosa.display.specshow(
        cqt, x_axis='s', y_axis='cqt_note', sr=22050, hop_length=256,
        fmin=librosa.note_to_hz('C1'), bins_per_octave=60)
    plt.title('Cqt of song')
    plt.set_cmap('binary')

    plt.subplot(313)
    librosa.display.specshow(
        cqt + 5 * f0_arr, x_axis='s', y_axis='cqt_note', sr=22050, hop_length=256,
        fmin=librosa.note_to_hz('C1'), bins_per_octave=60)
    plt.title('f0 overlayed on CQT')
    plt.set_cmap('binary')

    plt.tight_layout()
    plt.show()


def artificial_piano(f, sr, length, phis=None):
    H = (1, 3, 5)
    if phis is None:
        phis = {h: -np.pi / 2 for h in H}

    # add 2, 3, 4, 8 - th subharmonics to enrich sound
    piano_sound = 0
    for h in H:
        piano_sound += 1 / h * librosa.tone(h * f, sr=sr, length=length, phi=phis[h])

    # normalize value
    piano_sound /= sum([1 / h for h in H])

    def get_phi(f, phi):
        return 2 * np.pi * f * length / sr + phi

    phis = {h: get_phi(h * f, phis[h]) for h in H}
    # put some decay
    return piano_sound, phis


def smooth_f0_midi(f0_midi, window_len: int = 11):
    padded = np.pad(f0_midi, (window_len // 2, window_len // 2), 'constant',
                    constant_values=(-1, -1))
    smoothed = np.zeros(f0_midi.shape)
    for i in range(len(smoothed)):
        windowed_arr = padded[i: i + window_len]
        most_comm_val = collections.Counter(windowed_arr).most_common()[0][0]
        most_comm_count = collections.Counter(windowed_arr).most_common()[0][1]
        if most_comm_val == -1 and most_comm_count > window_len * 0.3:
            smoothed[i] = most_comm_val
        else:
            smoothed[i] = np.mean(windowed_arr[windowed_arr != -1])

    return smoothed


def midi_to_pcm(midi_smoothed: np.ndarray,
                cqt_config: CQTconfig = CQTconfig()) -> np.ndarray:
    stack = []
    phi = None
    for idx, midi in enumerate(midi_smoothed):
        if midi == -1:
            if idx == 0:
                audio_chunk = np.array([0 for _ in range(cqt_config.hop_length)])
            else:
                amplitude = stack[-1][-1]  # decay from latest pcm point
                audio_chunk = amplitude * np.exp(
                    - 2 * np.arange(cqt_config.hop_length) / cqt_config.hop_length)
            stack.append(audio_chunk)
        else:
            f = librosa.midi_to_hz(midi)

            piano_sound, phi = artificial_piano(
                f, cqt_config.sr, cqt_config.hop_length, phi)

            stack.append(piano_sound)

    pcm = np.concatenate(stack)
    return pcm


def load_audio(afile_info: AudioFileInfo, cqt_config: CQTconfig = CQTconfig()):
    duration = afile_info.end_t - afile_info.start_t
    y, _ = librosa.load(afile_info.audio_path,
                        sr=cqt_config.sr,
                        offset=afile_info.start_t,
                        duration=duration)
    return y


if __name__ == '__main__':
    print(hz_to_cqt_bin(134, 'C1', 60))

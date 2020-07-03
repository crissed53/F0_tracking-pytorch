import librosa
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple


__all__ = ['CQTconfig', 'gen_hcqt', 'hz_to_cqt_bin']


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


def gen_hcqt(y: np.ndarray, h_factor: Tuple[float],
             cqt_config: CQTconfig) -> np.ndarray:
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


if __name__ == '__main__':
    print(hz_to_cqt_bin(134, 'C1', 60))
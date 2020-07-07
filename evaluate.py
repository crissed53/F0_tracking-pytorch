import IPython.display as ipd
import torch
import torch.nn as nn
import tqdm

from dataset import F0TrackingDataset
from model.model import SalomonF0Tracker
from utils.feature_utils import *
import librosa.display


def load_model(device: torch.device = torch.device('cpu'),
               model_dir: str = './save/models/model_090.pt'):
    """
    Load NN model
    Args:
        net: neural network
        device: device to load features and outputs into
        model_dir: directory for neural model

    Returns:
        Loaded NN model

    """
    net = SalomonF0Tracker()
    state_dict = torch.load(model_dir, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def get_f0_from_hcqt(net: nn.Module, hcqt: np.ndarray,
                     expand_dim: bool = True) -> np.ndarray:
    """
    Convert hcqt to f0 using trained neural network
    Args:
        net: trained neural network
        hcqt: hcqt
        expand_dim: if True, add dimension in axis=0 for batch dimension

    Returns:
        f0 array obtained from the neural network

    """
    if expand_dim:
        hcqt = hcqt[np.newaxis, ...]

    output: torch.Tensor = net(torch.from_numpy(hcqt).float())

    return output.detach().numpy()


def get_hcqt_from_audio(audio_path: str, start_t: float, end_t: float,
                        cqt_config: CQTconfig = CQTconfig()) -> np.ndarray:
    """
    Get hcqt from audio. Audio is loaded from file by specifying its timestamps
    and its filepath
    Args:
        audio_path: file path of audio
        start_t: starting time in second
        end_t: end time in second
        cqt_config: config used to generate CQT

    Returns:
        hcqt array

    """
    y = load_audio(audio_path, start_t, end_t, cqt_config.sr)
    hcqt = np.abs(gen_hcqt(y))
    return hcqt


def load_audio(audio_path: str, start_t: float, end_t: float,
               sr: int = 22050) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=sr, offset=start_t, duration=end_t-start_t)
    return y


def gen_f0_from_audio(
        audio_path: str, start_t: float, end_t: float, net: nn.Module):
    frame_len_feature = F0TrackingDataset.FEATURE_LEN
    hcqt = get_hcqt_from_audio(audio_path, start_t, end_t)
    hcqt_frame_len = hcqt.shape[-1]

    if hcqt_frame_len < frame_len_feature:
        raise ValueError(f'Number of hcqt frames too small. Expects value '
                         f'larger than {frame_len_feature} but got {hcqt_frame_len}')

    num_features = hcqt_frame_len // frame_len_feature
    residue = hcqt_frame_len - num_features * frame_len_feature

    stack = []
    for i in range(num_features):
        f0_hat = get_f0_from_hcqt(
            net, hcqt[:, :, i * frame_len_feature: (i + 1) * frame_len_feature])
        stack.append(f0_hat)

    f0 = (np.concatenate(stack, axis=-1) > 0.5).astype(int)
    f0 = f0[0][0]  # discard dim for batch and harmonics

    f0_audio = f0
    cqt = hcqt[1, :, :-residue]

    return {'f0': f0, 'cqt': cqt}


def get_sine_wave_from_f0(f0: np.ndarray, cqt_config: CQTconfig = CQTconfig()):
    bin_num = np.argmax(f0, axis=0)
    bin_num = [val if val != 0 else -1 for val in bin_num]
    stack = []
    phi = 0
    for bin in bin_num:
        if bin == -1:
            audio_chunk = np.array([0 for _ in range(cqt_config.hop_length)])
            stack.append(audio_chunk)
            phi = 0
        else:
            f = cqt_bin_to_hz(bin, cqt_config.fmin_note,
                              cqt_config.bins_per_octave)
            audio_chunk = librosa.tone(
                f, sr=cqt_config.sr, length=cqt_config.hop_length, phi=phi)
            # save phase information of the latest audio chunk in order
            # for smooth concat btw audio chunks
            stack.append(audio_chunk)
            phi = 2 * np.pi * f * cqt_config.hop_length / cqt_config.sr

    f0_audio = np.concatenate(stack)

    return np.concatenate(stack)


def audio_to_cqt_and_f0(audio_path: str, start_t: int, end_t: int, net: nn.Module,
                        frame_hop: int = 25, thres: float = 0.25) -> dict:
    frame_len_feature = F0TrackingDataset.FEATURE_LEN

    num_overlap = frame_len_feature // frame_hop

    hcqt = get_hcqt_from_audio(audio_path, start_t, end_t)
    hcqt_frame_len = hcqt.shape[-1]

    if hcqt_frame_len < frame_len_feature:
        raise ValueError(f'Number of hcqt frames too small. Expects value '
                         f'larger than {frame_len_feature} but got {hcqt_frame_len}')

    num_features = (hcqt_frame_len - frame_len_feature) // frame_hop + 1
    f_hat_total = np.zeros(hcqt.shape[-2:])
    prog_bar = tqdm.tqdm(total=num_features, position=0)
    for i in range(num_features):
        start_frame = i * frame_hop
        end_frame = start_frame + frame_len_feature
        f0_hat = get_f0_from_hcqt(net, hcqt[:, :, start_frame: end_frame])
        f_hat_total[:, start_frame: end_frame] += f0_hat[0][0]
        prog_bar.update()

    f0 = (f_hat_total > thres * num_overlap).astype(int)
    cqt = hcqt[1, :, :]

    return {'f0': f0, 'cqt': cqt}


def f0_tracking_demo(audio_path: str, start_t: int, end_t: int,
                     frame_hop: int = 25, interactive: bool = True,
                     cqt_config: CQTconfig = CQTconfig(),
                     thres: float = 0.25, window_len: int = 11):

    afile_info = AudioFileInfo(audio_path, start_t, end_t)
    net = load_model()

    result = audio_to_cqt_and_f0(
        audio_path, start_t, end_t, net, frame_hop, thres)
    f0, cqt = result['f0'], result['cqt']

    midis = f0_contour_to_midi_contour(f0)
    midis_smoothed = smooth_f0_midi(midis, window_len)

    if interactive:
        # visualize f0 and cqt
        visualize_f0_and_cqt(f0, cqt, afile_info)

        # visualize f0 contour, in midi number, both raw and smoothed version
        plt.figure(figsize=(20, 10))
        plt.title('F0 contour, raw vs smoothed')
        plt.plot(midis, linewidth=1, label='Raw f0 Contour')
        plt.plot(midis_smoothed, '--', label='Smoothed f0 contour')
        plt.show()

        f0_audio = midi_to_pcm(midis_smoothed)

        original_audio = load_audio(audio_path, start_t, end_t)
        print('Original')
        ipd.display(ipd.Audio(original_audio, rate=cqt_config.sr))
        print('Generated f0')
        ipd.display(ipd.Audio(f0_audio, rate=cqt_config.sr))


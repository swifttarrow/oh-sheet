# Copyright 2022 The MT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Audio spectrogram functions."""

import dataclasses

# for PyTorch spectrogram
import torch
from torchaudio.transforms import MelSpectrogram
import librosa
import numpy as np

# this is to suppress a warning from torch melspectrogram
import warnings
warnings.filterwarnings("ignore")

# for TF spectrogram (optional — only needed if use_tf_spectral_ops=True)
try:
    from ddsp import spectral_ops
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except ImportError:
    spectral_ops = None
    tf = None

# defaults for spectrogram config
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_HOP_WIDTH = 128
DEFAULT_NUM_MEL_BINS = 512

# fixed constants; add these to SpectrogramConfig before changing
FFT_SIZE = 2048
MEL_LO_HZ = 20.0


@dataclasses.dataclass
class SpectrogramConfig:
    """Spectrogram configuration parameters."""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    hop_width: int = DEFAULT_HOP_WIDTH
    num_mel_bins: int = DEFAULT_NUM_MEL_BINS
    use_tf_spectral_ops: bool = False

    @property
    def abbrev_str(self):
        s = ''
        if self.sample_rate != DEFAULT_SAMPLE_RATE:
            s += 'sr%d' % self.sample_rate
        if self.hop_width != DEFAULT_HOP_WIDTH:
            s += 'hw%d' % self.hop_width
        if self.num_mel_bins != DEFAULT_NUM_MEL_BINS:
            s += 'mb%d' % self.num_mel_bins
        return s

    @property
    def frames_per_second(self):
        return self.sample_rate / self.hop_width


def split_audio(samples, spectrogram_config):
    """Split audio into frames."""
    if spectrogram_config.use_tf_spectral_ops:
        # print("split TF")
        return tf.signal.frame(
            samples,
            frame_length=spectrogram_config.hop_width,
            frame_step=spectrogram_config.hop_width,
            pad_end=True)
    else:
        # print("split PT")
        if samples.shape[0] % spectrogram_config.hop_width != 0:
            samples = np.pad(
                samples, 
                (0, spectrogram_config.hop_width - samples.shape[0] % spectrogram_config.hop_width), 
                'constant',
                constant_values=0
            )
        return librosa.util.frame(
            samples,
            frame_length=spectrogram_config.hop_width,
            hop_length=spectrogram_config.hop_width,
            axis=-1).T

def pad_end(samples, n_fft, hop_size):
    """Pad the waveform to ensure that all samples are processed."""
    n_samples = samples.shape[-1]
    # using double negatives to round up
    n_frames = -(-n_samples // hop_size)
    pad_samples = max(0, n_fft + hop_size * (n_frames - 1) - n_samples)
    return torch.nn.functional.pad(samples, (0, pad_samples))

def safe_log(x, eps=1e-5):
    """Avoid taking the log of a non-positive number."""
    safe_x = torch.where(x <= 0.0, eps, x)
    return torch.log(safe_x)

def compute_spectrogram(
    samples, 
    spectrogram_config,
):
    """
    Compute a mel spectrogram.
    Due to multiprocessing issues running TF and PyTorch together, we use librosa
    and only keep `spectral_ops.compute_logmel` for evaluation purposes.
    """
    if spectrogram_config.use_tf_spectral_ops:
        # NOTE: we only keep this for evaluating existing models
        # This is because I find even with an equivalent PyTorch / librosa implementation 
        # that gives close-enough results (melspec MAE ~ 2e-3), the model output is still affected badly.
        # lazy load
        # print("spec TF")
        overlap = 1 - (spectrogram_config.hop_width / FFT_SIZE)
        return spectral_ops.compute_logmel(
            samples,
            bins=spectrogram_config.num_mel_bins,
            lo_hz=MEL_LO_HZ,
            overlap=overlap,
            fft_size=FFT_SIZE,
            sample_rate=spectrogram_config.sample_rate)
    else:
        # print("spec PT")
        transform = MelSpectrogram(
            sample_rate=spectrogram_config.sample_rate,
            n_fft=FFT_SIZE,
            hop_length=spectrogram_config.hop_width,
            n_mels=spectrogram_config.num_mel_bins,
            f_min=MEL_LO_HZ,
            f_max=7600,
            power=1.0,
            center=False
        )
        samples = torch.from_numpy(samples).float()
        S = transform(pad_end(samples, FFT_SIZE, spectrogram_config.hop_width))
        S = safe_log(S)
        # S[S<0] = 0
        # S = torch.log(S + 1e-6)
        return S.numpy().T


def flatten_frames(frames, use_tf_spectral_ops=False):
    """Convert frames back into a flat array of samples."""
    if use_tf_spectral_ops:
        # print("flatten TF")
        return tf.reshape(frames, (-1,))
    else:
        # print("flatten PT")
        return np.reshape(frames, (-1,))


def input_depth(spectrogram_config):
    return spectrogram_config.num_mel_bins

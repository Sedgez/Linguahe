"""Shared audio feature extraction for the emotion ANN."""

from math import gcd, log
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.signal import resample_poly


SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128


def _hz_to_mel(hz):
    hz = np.asarray(hz, dtype=float)
    mel = hz / (200.0 / 3)
    mask = hz >= 1000
    mel[mask] = 15 + np.log(hz[mask] / 1000) / (log(6.4) / 27)
    return mel


def _mel_to_hz(mel):
    mel = np.asarray(mel, dtype=float)
    hz = mel * (200.0 / 3)
    mask = mel >= 15
    hz[mask] = 1000 * np.exp((log(6.4) / 27) * (mel[mask] - 15))
    return hz


def _make_mel_filterbank():
    frequencies = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
    mel_points = np.linspace(
        _hz_to_mel(np.array([0.0]))[0],
        _hz_to_mel(np.array([SAMPLE_RATE / 2]))[0],
        N_MELS + 2,
    )
    hz_points = _mel_to_hz(mel_points)
    filters = np.zeros((N_MELS, len(frequencies)), dtype=np.float64)

    for index in range(N_MELS):
        left, center, right = hz_points[index:index + 3]
        filters[index] = np.maximum(
            0,
            np.minimum(
                (frequencies - left) / (center - left),
                (right - frequencies) / (right - center),
            ),
        )
        filters[index] *= 2.0 / (right - left)

    return frequencies, filters


_FREQUENCIES, _MEL_FILTERBANK = _make_mel_filterbank()
_WINDOW = np.hanning(N_FFT + 1)[:-1]


def load_emotion_audio(path, target_sr=SAMPLE_RATE):
    """Load a mono WAV and resample it to the ANN's expected rate."""
    signal, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    return resample_emotion_audio(signal, sample_rate, target_sr)


def resample_emotion_audio(signal, sample_rate, target_sr=SAMPLE_RATE):
    """Convert a waveform to mono and resample it for emotion features."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if signal.size == 0:
        raise ValueError("Cannot extract emotion features from empty audio.")

    if sample_rate != target_sr:
        divisor = gcd(int(sample_rate), int(target_sr))
        signal = resample_poly(
            signal,
            target_sr // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32)

    return signal


def extract_emotion_features(signal, sample_rate=SAMPLE_RATE):
    """Return the 18 ANN features in the same order used for training."""
    signal = resample_emotion_audio(signal, sample_rate)
    padding = N_FFT // 2
    padded = np.pad(
        signal,
        (padding, padding),
        mode="reflect" if len(signal) > 1 else "constant",
    )
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH]
    spectrum = np.fft.rfft(frames * _WINDOW, axis=1).T
    magnitude = np.abs(spectrum)
    power = magnitude * magnitude

    mel_power = _MEL_FILTERBANK @ power
    decibels = 10 * np.log10(np.maximum(mel_power, 1e-10))
    decibels = np.maximum(decibels, decibels.max() - 80)
    mfcc = dct(decibels, type=2, axis=0, norm="ortho")[:13].mean(axis=1)

    zcr = np.mean(
        np.count_nonzero(np.diff(frames < 0, axis=1), axis=1) / N_FFT
    )
    rms = np.mean(np.sqrt(np.mean(frames * frames, axis=1)))
    magnitude_sum = np.maximum(magnitude.sum(axis=0), 1e-10)
    centroid = (_FREQUENCIES[:, None] * magnitude).sum(axis=0) / magnitude_sum
    bandwidth = np.sqrt(
        (
            magnitude
            * np.abs(_FREQUENCIES[:, None] - centroid[None, :]) ** 2
        ).sum(axis=0)
        / magnitude_sum
    )
    cumulative = np.cumsum(magnitude, axis=0)
    rolloff_indices = (
        cumulative >= 0.85 * magnitude_sum[None, :]
    ).argmax(axis=0)
    rolloff = _FREQUENCIES[rolloff_indices]

    return np.asarray(
        [
            *mfcc,
            zcr,
            rms,
            centroid.mean(),
            bandwidth.mean(),
            rolloff.mean(),
        ],
        dtype=np.float32,
    )


def extract_emotion_features_from_file(path):
    signal = load_emotion_audio(Path(path))
    return extract_emotion_features(signal)

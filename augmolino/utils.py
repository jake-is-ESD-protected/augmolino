# ------[augmolino]------
#   @ name: augmolino.utils
#   @ auth: Jakob Tschavoll
#   @ vers: 0.1
#   @ date: 2022


"""
Utility functions for WAV-files and augmolino-specific methods
"""

import matplotlib.pyplot as plt
import librosa.display
import librosa as lr

__all__ = ['spectrogram']



def spectrogram(signal, _sr=22050):
    """
    Draw a spectrogram of the specified signal

    Params
    ------
    `signal`:   Audio data in array format
    `_sr`:      Desired sample rate of audio

    Returns
    -------
    `fp_dest`:      redundant path `fp_dest`
    """

    try:
        x, sr = lr.load(signal)
    except TypeError:
        x = signal
        sr = _sr
    print(sr)
    X = lr.stft(x)
    Xdb = lr.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
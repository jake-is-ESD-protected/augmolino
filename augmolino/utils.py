import matplotlib.pyplot as plt
import librosa.display
import librosa as lr

# render spectrogram of a stored audio-file
def spectrogram(signal, _sr=22050):
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
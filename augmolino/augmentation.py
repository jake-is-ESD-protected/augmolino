# needed imports
import librosa as lr
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# render spectrogram of a stored audio-file
def spectrogram(fp_source):
    x, sr = lr.load(fp_source)
    X = lr.stft(x)
    Xdb = lr.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()


# stretch or squeeze file while retaining pitch
# factor < 0: squeeze
# factor > 0: stretch
def timeStretch(fp_source, fp_dest, factor):
    x, sr = lr.load(fp_source)
    x_new = lr.effects.time_stretch(x, factor)
    sf.write(fp_dest, x_new, sr)
    return fp_dest


# shift pitch of file
# factor < 0: pitch down
# factor > 0: pitch up
def pitchShift(fp_source, fp_dest, factor):
    x, sr = lr.load(fp_source)
    x_new = lr.effects.pitch_shift(x, sr, factor)
    sf.write(fp_dest, x_new, sr)
    return fp_dest


# introduce dead-time to start or skip start by seconds
# s < 0: skip first few seconds
# s > 0: add dead-time to start
def offsetAudio(fp_source, fp_dest, s):
    x, sr = lr.load(fp_source)
    sample_offs = sr * abs(s)
    if len(x) <= sample_offs:
        return None
    if s < 0:
        x = np.copy(x[sample_offs: ])
    else:
        for i in range(sample_offs):
            x = np.insert(x, 0, 0)  
    sf.write(fp_dest, x, sr)        
    return fp_dest


# create fade-in/fade-out
# direction = "out": fadeout
# direction = "in": fadein
def fadeAudio(fp_source, fp_dest, s, direction):
    x, sr = lr.load(fp_source)
    fade_len = sr * s
    if direction == "out":   
        end = x.shape[0]
        start = end - fade_len
        fade_curve = np.logspace(0, -3, fade_len)
        x[start:end] = x[start:end] * fade_curve
    elif direction == "in":
        fade_curve = np.logspace(-3, 0, fade_len)
        x[0:fade_len] = x[0:fade_len] * fade_curve
    else:
        return None
    sf.write(fp_dest, x, sr)        
    return fp_dest
     

# merge two sounds
# lvl_ratio: 0 <= value <= 1, 1 ignores the noise, 0 the main sound
def fuseAudio(fp_source_sound, fp_source_noise, fp_dest, lvl_ratio = 0.5):
    snd, sr_snd = lr.load(fp_source_sound)
    noise, sr_noise = lr.load(fp_source_noise, sr=sr_snd)
    x_new = np.zeros(shape=(len(snd)))
    
    for i in range(len(snd)):
        x_new[i] = snd[i]*(lvl_ratio) + noise[i]*(1 - lvl_ratio)
    
    sf.write(fp_dest, x_new, sr_snd)        
    return fp_dest
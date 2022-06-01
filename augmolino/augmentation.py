# ------[augmolino]------
#   @ name: augmolino.augmentation
#   @ auth: Jakob Tschavoll
#   @ vers: 0.1
#   @ date: 2022

"""
Various WAV-based augmentation methods. Each method adds an augmented copy of the file, 
specified by `fp_source` and `fp_dest`
"""


import librosa as lr
import numpy as np
import soundfile as sf
import random as rd

__all__ = ['timeStretch', 'pitchShift', 'offsetAudio', 'fadeAudio', 'fuseAudio']



def timeStretch(fp_source, fp_dest, factor):
    """
    Stretch or squeeze a WAV-file while retaining pitch.

    Params
    ------
    `fp_source`:    File path to file to be augmented
    `fp_dest`:      Path and name of augmented file (augmented copy)
    `factor`:       Factor by which time is stretched (< 1 means shorter, > 1 means longer)

    Returns
    -------
    `fp_dest`:      redundant path `fp_dest`
    """

    x, sr = lr.load(fp_source)
    x_new = lr.effects.time_stretch(x, factor)
    sf.write(fp_dest, x_new, sr)
    return fp_dest



def pitchShift(fp_source, fp_dest, factor):
    """
    Pitch-shift a WAV-file while retaining length.

    Params
    ------
    `fp_source`:    File path to file to be augmented
    `fp_dest`:      Path and name of augmented file (augmented copy)
    `factor`:       Factor by which pitch is modified (< 1 means lower, > 1 means higher)

    Returns
    -------
    `fp_dest`:      redundant path `fp_dest`
    """

    x, sr = lr.load(fp_source)
    x_new = lr.effects.pitch_shift(x, sr, factor)
    sf.write(fp_dest, x_new, sr)
    return fp_dest



def offsetAudio(fp_source, fp_dest, s):
    """
    Offset a WAV-file by added dead-time or or by later start

    Params
    ------
    `fp_source`:    File path to file to be augmented
    `fp_dest`:      Path and name of augmented file (augmented copy)
    `s`:            Offset in seconds. s < 0: skip first few seconds. s > 0: add dead-time to start

    Returns
    -------
    `fp_dest`:      redundant path `fp_dest`
    """

    x, sr = lr.load(fp_source)
    sample_offs = int(sr * abs(s))
    if len(x) <= sample_offs:
        return None
    if s < 0:
        x = np.copy(x[sample_offs: ])
    else:
        for i in range(sample_offs):
            x = np.insert(x, 0, 0)  
    sf.write(fp_dest, x, sr)        
    return fp_dest



def fadeAudio(fp_source, fp_dest, s, direction="in"):
    """
    Create fade-in or fade-out for a WAV-file.

    Params
    ------
    `fp_source`:    File path to file to be augmented
    `fp_dest`:      Path and name of augmented file (augmented copy)
    `s`:            Fade-time in seconds.
    `direction:`    Direction to fade from. Direction = "out": fadeout. Direction = "in": fadein.

    Returns
    -------
    `fp_dest`:      redundant path `fp_dest`
    """

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
     

def fuseAudio(fp_source_sound, fp_source_noise, fp_dest, lvl_ratio = 0.5):
    """
    Mix a WAV-file with a specified noisy WAV-file at a random timestamp

    Params
    ------
    `fp_source_sound`:  File path to file to be augmented
    `fp_source_noise`:  File path of noise to be mixed in
    `fp_dest`:          Path and name of augmented file (augmented copy)
    `lvl_ratio`:        Ratio by which the sounds are mixed. 0 <= value <= 1, 1 ignores the noise, 0 the main sound.

    Returns
    -------
    `fp_dest`:      redundant path `fp_dest`
    `rand_start`:   timestamp of noise-file where the mixing started

    Notes
    -----
    Augmented file is as long as the original sound of interest, not the mixed-in sound
    """
    
    snd, sr_snd = lr.load(fp_source_sound)
    noise, sr_noise = lr.load(fp_source_noise, sr=sr_snd)
    rd_value = int(1000*snd[int(len(snd)/2)])
    rd.seed(rd_value) # use value of center sample as seed
    rand_start = rd.randint(0, len(noise)-len(snd))
    
    if rand_start < 0:
        return None
    rand_part_noise = np.copy(noise[rand_start:(rand_start+len(snd))])
    x_new = np.zeros(shape=(len(snd)))
    
    for i in range(len(snd)):
        x_new[i] = snd[i]*(lvl_ratio) + rand_part_noise[i]*(1 - lvl_ratio)
    
    sf.write(fp_dest, x_new, sr_snd)        
    return fp_dest, (rand_start/sr_snd)
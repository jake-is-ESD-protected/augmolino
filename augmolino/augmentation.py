import warnings
import librosa as lr
import numpy as np
import soundfile as sf
import random as rd
import os

__all__ = ['timeStretch', 'pitchShift',
           'offsetAudio', 'fadeAudio', 'mixAudio']

descriptors = {
    "_augmentation": "base class",
    __all__[0]: "time_stretch",
    __all__[1]: "pitch_shift",
    __all__[2]: "time_offset",
    __all__[3]: "time_fade",
    __all__[4]: "sound_mix"}


class _augmentation:
    """
    Private base class for different augmentations.
    Prepares general parameters which every augmentation
    needs. 

    Parameters
    ----------
    `f_source`:
        path-like string to file which should be augmented
    `f_dest`:
        path-like string to preferred storage location or name
    `sample_rate`:
        sample rate of audio file (used for resampling)    

    Notes
    -----
    This is a private base class which is not intened to be
    called on its own. Always call an actual augmentation.
    """

    def __init__(self, f_source, f_dest=None, sample_rate=22050):

        path_details = os.path.splitext(f_source)
        extension = path_details[1]
        self.auto_name = False

        if extension != ".wav":
            raise ValueError("File type not supported! Use '.wav' instead.")

        self.f_source = f_source
        self.sample_rate = sample_rate
        self.descriptor = descriptors["_augmentation"]

        # user wants to save the augmentation
        if f_dest != None:
            if f_source == f_dest:
                warnings.warn("Source and save name are the same,\
                    original file will be overwritten!")
                self.f_dest = f_dest
            elif f_dest == "auto":
                # name target file later in specific augmentation
                self.auto_name = True
                self.f_dest = None
            else:
                self.f_dest = f_dest

        # user wants a temporary array
        else:
            self.f_dest = None

    def load(self):

        x, sr = lr.load(path=self.f_source, sr=self.sample_rate)
        self.signal = x

    def _autoName(self, descriptor, param):
        self.f_dest = self.f_source[:-4] + f"_{descriptor}_{param}.wav"


class timeStretch(_augmentation):
    """
    Stretch or squeeze a sound while retaining pitch.

    Parameters
    ----------
    `f_source`:
        String. Path-like string to file which should be augmented
    `f_dest`:
        String. Path-like string to preferred storage location or name.
        Default is `None`
    `rate`:
        Float. Stretch factor defined by speed of playback. A `rate` < 1
        stretches the sound, a `rate` > 1 squeezes it.
        Default is `1`
    `sample_rate`:
        Int. Sample rate of audio file (used for resampling).
        Default is `22050`

    Returns
    -------
    `None`:
        If `f_dest` was set
    `x`:
        Array. Numpy array containing augmented signal
    """

    def __init__(self, f_source,
                 f_dest=None, rate=1, sample_rate=22050):
        super().__init__(f_source=f_source,
                         f_dest=f_dest, sample_rate=sample_rate)
        self.rate = rate
        self.descriptor = descriptors[__all__[0]]

        if self.auto_name:
            self._autoName(self.descriptor, rate)

    def run(self):

        self.load()
        x_new = lr.effects.time_stretch(y=self.signal, rate=self.rate)

        if self.f_dest != None:
            sf.write(self.f_dest, x_new, self.sample_rate)
            return None

        else:
            return x_new


class pitchShift(_augmentation):
    """
    Shift a given input signal's pitch by semitones.

    Parameters
    ----------
    `f_source`:
        String. Path-like string to file which should be augmented
    `f_dest`:
        String. Path-like string to preferred storage location or name.
        Default is `None`
    `semitones`:
        Float. Number of western scale semitones to shift up or down.
        Default is `1`
    `sample_rate`:
        Int. Sample rate of audio file (used for resampling).
        Default is `22050`

    Returns
    -------
    `None`:
        If `f_dest` was set
    `x`:
        Array. Numpy array containing augmented signal
    """

    def __init__(self, f_source,
                 f_dest=None, semitones=1, sample_rate=22050):
        super().__init__(f_source=f_source,
                         f_dest=f_dest, sample_rate=sample_rate)
        self.semitones = semitones
        self.descriptor = descriptors[__all__[1]]

        if self.auto_name:
            self._autoName(self.descriptor, semitones)

    def run(self):

        self.load()
        x_new = lr.effects.pitch_shift(y=self.signal,
                                       sr=self.sample_rate, n_steps=self.semitones)

        if self.f_dest != None:
            sf.write(self.f_dest, x_new, self.sample_rate)
            return None

        else:
            return x_new


class offsetAudio(_augmentation):
    """
    Offset a sound by added dead-time or or by later start.

    Parameters
    ----------
    `f_source`:
        String. Path-like string to file which should be augmented
    `f_dest`:
        String. Path-like string to preferred storage location or name.
        Default is `None`
    `s`:
        Float. Offset in seconds. s < 0: skip first samples. 
        s > 0: add dead-time to start. Default is `0`.
    `sample_rate`:
        Int. Sample rate of audio file (used for resampling).
        Default is `22050`

    Returns
    -------
    `None`:
        If `f_dest` was set
    `x`:
        Array. Numpy array containing augmented signal
    """

    def __init__(self, f_source,
                 f_dest=None, s=0, sample_rate=22050):
        super().__init__(f_source=f_source,
                         f_dest=f_dest, sample_rate=sample_rate)
        self.s = s
        self.descriptor = descriptors[__all__[2]]

        if self.auto_name:
            self._autoName(self.descriptor, s)

    def run(self):

        self.load()
        sample_offs = int(self.sample_rate * abs(self.s))

        if len(self.signal) <= sample_offs:
            raise ValueError("Offset longer than duration of sound!")
        if self.s < 0:
            x_new = self.signal[sample_offs:]
        else:
            x_new = np.insert(self.signal, 0, np.zeros(sample_offs))

        if self.f_dest != None:
            sf.write(self.f_dest, x_new, self.sample_rate)
            return None

        else:
            return x_new


class fadeAudio(_augmentation):
    """
    Create a logarithmic fade-in or fade-out for a sound.

    Parameters
    ----------
    `f_source`:
        String. Path-like string to file which should be augmented
    `f_dest`:
        String. Path-like string to preferred storage location or name.
        Default is `None`
    `s`:
        Float. Fade time in seconds. Default is `0`.
    `direction`:
        String. Direction from which the fade is applied. Default
        is `"in"`.    
    `sample_rate`:
        Int. Sample rate of audio file (used for resampling).
        Default is `22050`

    Returns
    -------
    `None`:
        If `f_dest` was set
    `x`:
        Array. Numpy array containing augmented signal
    """

    def __init__(self, f_source, f_dest=None, s=0,
                 direction="in", sample_rate=22050):

        if direction not in ["in", "out"]:
            raise ValueError(f"parameter '{direction}' not recognized!")

        super().__init__(f_source=f_source,
                         f_dest=f_dest, sample_rate=sample_rate)
        self.s = s
        self.direction = direction
        self.descriptor = descriptors[__all__[3]]

        if self.auto_name:
            self._autoName(self.descriptor, str(s) + f"_{direction}")

    def run(self):

        self.load()
        fade_len = self.sample_rate * self.s
        x_new = self.signal

        if self.direction == "out":
            end = len(self.signal)
            start = end - fade_len
            fade_curve = np.logspace(0, -3, fade_len)
            x_new[start:end] *= fade_curve
        else:
            fade_curve = np.logspace(-3, 0, fade_len)
            x_new[0:fade_len] *= fade_curve

        if self.f_dest != None:
            sf.write(self.f_dest, x_new, self.sample_rate)
            return None

        else:
            return x_new


class mixAudio(_augmentation):
    """
    Mix two wavefiles at random or specified timestamps.

    Parameters
    ----------
    `f_source`:
        String. Path-like string to file which should be augmented
    `f_source_mix`:
        String. Path-like string to file which should be mixed in.
    `f_dest`:
        String. Path-like string to preferred storage location or name.
        Default is `None`
    `ratio`:
        Float. Ratio by which the sounds are mixed. 
        `0 <= ratio <= 1`, 1 ignores the noise, 0 the main sound. 
        Default is `1`.
    `start_at`:
        Float. Second at which the mixed in sound should be used.
        If unspecified, a random time will be used. Default is `None`.     
    `sample_rate`:
        Int. Sample rate of audio file (used for resampling).
        Default is `22050`

    Returns
    -------
    `None`:
        If `f_dest` was set
    `x`:
        Array. Numpy array containing augmented signal

    Notes
    -----
    Augmented sound is as long as the original sound of interest, 
    not the mixed-in sound    
    """

    def __init__(self, f_source, f_source_mix, f_dest=None,
                 ratio=0.5, start_at=None, sample_rate=22050):

        super().__init__(f_source=f_source,
                         f_dest=f_dest, sample_rate=sample_rate)
        self.f_source_mix = f_source_mix
        self.ratio = ratio
        self.start_at = start_at
        self.descriptor = descriptors[__all__[4]]

        if self.auto_name:
            self._autoName(self.descriptor, ratio)

    def run(self):

        self.load()
        noise, _ = lr.load(path=self.f_source_mix,
                           sr=self.sample_rate)
        if self.start_at == None:
            # use value of center sample as seed
            rd_value = int(1000*self.signal[int(len(self.signal)/2)])
            rd.seed(rd_value)
            start = rd.randint(0, len(noise)-len(self.signal))
        else:
            start = int(self.start_at * self.sample_rate)

        part_noise = noise[start:(start+len(self.signal))]

        x_new = self.signal * self.ratio + part_noise * (1 - self.ratio)

        if self.f_dest != None:
            sf.write(self.f_dest, x_new, self.sample_rate)
            return None

        else:
            return x_new

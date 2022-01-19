# augmolino

`augmolino` is a small data-augmentation python module for data science and neural networks with audio-focus. Its methods are very file-based and user friendly for simple mass-augmentation.

---

## First things first!

- This module is for `wav`-files only
- Data augmentation needs huge amounts of memory
- Use this module to expand your data-sets

### Based on:

- [librosa](https://librosa.org/)
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)
- [soundfile](https://pypi.org/project/SoundFile/)

All methods operate on the same I/O logic:

- pass a `path-like` object of the source file
- pass a `path-like` object of the resulting file (doesn't need to exist yet!)
- pass a parameter specific to the augmentation

---

## Methods

```python
def timeStretch(fp_source, fp_dest, factor)
```

> Returns redundant path `fp_dest`

Stretch copy of source file by a factor around `1` and save it to `fp_dest`. Pitch remains unchanged.

<br />
<br />

```python
def picthShift(fp_source, fp_dest, factor)
```

> Returns redundant path `fp_dest`

Shift copy of source file by a factor of semitones and save it to `fp_dest`. Duration remains unchanged.

<br />
<br />

```python
def offsetAudio(fp_source, fp_dest, s)
```

> Returns redundant path `fp_dest` or `None` if the file duration is shorter than `s`

Offset copy of source file by `s` seconds of and save it to `fp_dest`. If `s` is positive, the file will have a delay of `s` seconds in form of dead time. If `s` is negative, the `s` seconds will be cut off from the start and the file will be shorter.

<br />
<br />

```python
def fadeAudio(fp_source, fp_dest, s, direction)
```

> Returns redundant path `fp_dest`

Create a logarithmic fade for `s` seconds of the copied file. Use `direction="in"` or `direction="out"` to specify the location of the fade.

<br />
<br />

```python
def fuseAudio(fp_source_sound, fp_source_noise, fp_dest, lvl_ratio)
```

> Returns redundant path `fp_dest`

Mix `fp_source_sound` and `fp_source_noise` for the lenght of `fp_source_sound` by a `lvl_ratio` between `0` and `1`. This can be used to modify a complete dataset with artificial noise to make it more robust.

<br />
<br />

```python
def spectrogram(fp_source)
```

> Returns `None`

Helper function for quick display of spectrogram of `fp_source`.

<br />
<br />

---

## Examples

todo

# augmolino

|||
|-|-|
|`augmolino` is a small data-augmentation python module for data science and neural networks with audio-focus. Its methods are very file-based and user friendly for simple mass-augmentation.|<img src="GRAPHICS/augmolino_logo.png" alt="logo" width="300"/>|


---

## First things first!

- This module is for `wav`-files only
- Data augmentation needs huge amounts of memory
- Use this module to expand your datasets

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
import os
import librosa as lr
import augmolino.augmentation as aug
import numpy as np
import numpy.testing as npt

test_file = "tests/sounds/impulse_response.wav"


def test_augmentation_init():

    a = aug._augmentation(test_file)

    assert a.descriptor == aug.descriptors["_augmentation"]
    assert a.sample_rate == 22050
    assert a.f_dest == None


def test_augmentation_load():

    a = aug._augmentation(test_file)
    a.load()

    t, _ = lr.load(path=test_file)

    assert len(a.signal) == len(t)


def test_timeStretch_class():

    test_rate = 0.5
    a = aug.timeStretch(test_file, rate=test_rate)

    assert a.descriptor == aug.descriptors[aug.__all__[0]]

    x = a.run()

    assert len(x) == int(len(a.signal) / test_rate)


def test_pitchShift_class():

    test_semitones = 2
    a = aug.pitchShift(test_file, semitones=test_semitones)

    assert a.descriptor == aug.descriptors[aug.__all__[1]]

    x = a.run()

    assert len(x) == len(a.signal)


def test_offsetAudio_class():

    test_s = 1
    a = aug.offsetAudio(test_file, s=test_s)

    assert a.descriptor == aug.descriptors[aug.__all__[2]]

    x = a.run()

    assert len(x) == len(a.signal) + test_s * a.sample_rate
    npt.assert_allclose(x[0:test_s * a.sample_rate],
                        np.zeros(test_s * a.sample_rate))

    test_s = -1
    a = aug.offsetAudio(test_file, s=test_s)

    x = a.run()
    assert len(x) == len(a.signal) + test_s * a.sample_rate


def test_fadeAudio_class():

    test_s = 1
    test_direction = "in"
    a = aug.fadeAudio(test_file, s=test_s,
                      direction=test_direction)

    assert a.descriptor == aug.descriptors[aug.__all__[3]]

    x = a.run()


def test_save_file():

    path = "testfile.wav"
    a = aug.timeStretch(test_file, f_dest=path, rate=0.5)

    x = a.run()

    assert x == None
    assert os.path.exists(path)

    os.remove(path)


def test_save_file_auto():

    path = "auto"
    test_rate = 0.5
    a = aug.timeStretch(test_file, f_dest=path, rate=test_rate)

    x = a.run()

    assert x == None
    target_path = f"{test_file[:-4]}_{a.descriptor}_{test_rate}.wav"
    assert os.path.exists(target_path)

    os.remove(target_path)

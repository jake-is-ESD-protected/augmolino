import os
import librosa as lr
import augmolino.augmentation as aug

test_file = "tests/sounds/impulse_response.wav"


def test_augmentation_init():

    a = aug._augmentation(test_file)

    assert a.descriptor == aug.descriptors["_augmentation"]
    assert a.sample_rate == 22050
    assert a.f_dest == None


def test_augmentation_load():

    a = aug._augmentation(test_file)
    a.load()

    t, _ = lr.load(test_file)

    assert len(a.signal) == len(t)


def test_timeStretch_class():

    test_rate = 0.5
    a = aug.timeStretch(test_file, rate=test_rate)

    assert a.descriptor == aug.descriptors[aug.__all__[0]]

    x = a.run()

    assert len(x) == int(len(a.signal) / test_rate)

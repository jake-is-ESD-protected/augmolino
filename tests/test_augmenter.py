from augmolino import augmenter, augmentation
import numpy as np


def test_empty_augmenter_init():

    a = augmenter.augmenter()
    assert a.pipe == []


def test_filled_augmenter_init():

    augs = [
        augmentation.pitchShift(
            "tests/sounds/impulse_response.wav", semitones=1),
        augmentation.pitchShift(
            "tests/sounds/impulse_response.wav", semitones=2)]

    a = augmenter.augmenter(augs)

    assert len(a.pipe) == len(augs)

    for aug in a.pipe:
        assert aug.descriptor == "pitch_shift"
        assert aug.f_source != ""
        assert aug.f_source != None


def test_augmenter_summary():

    augs = [
        augmentation.pitchShift(
            "tests/sounds/impulse_response.wav", semitones=1),
        augmentation.pitchShift(
            "tests/sounds/impulse_response.wav", semitones=2)]

    a = augmenter.augmenter(augs)

    a.summary()


def test_augmenter_run_array():

    augs = [
        augmentation.timeStretch(
            "tests/sounds/impulse_response.wav", rate=2),
        augmentation.pitchShift(
            "tests/sounds/impulse_response.wav", semitones=2),
        augmentation.offsetAudio(
            "tests/sounds/impulse_response.wav", s=1)]

    a = augmenter.augmenter(augs)

    xs = a.execute()

    assert len(xs) == len(augs)


# this can only be tested visually by a human
test_augmenter_summary()

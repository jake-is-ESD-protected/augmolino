from augmolino import augmenter, augmentation


def test_empty_augmenter_init():

    a = augmenter.augmenter()
    assert a.pipe == []


def test_filled_augmenter_init():

    augs = [
        augmentation.pitchShift(
            "../sounds/impulse_response.wav", semitones=1),
        augmentation.pitchShift(
            "../sounds/impulse_response.wav", semitones=2)]

    a = augmenter.augmenter(augs)

    assert len(a.pipe) == len(augs)

    for aug in a.pipe:
        assert aug.descriptor == "pitch shift"
        assert aug.f_source != ""
        assert aug.f_source != None


def test_augmenter_summary():

    augs = [
        augmentation.pitchShift(
            "../sounds/impulse_response.wav", semitones=1),
        augmentation.pitchShift(
            "../sounds/impulse_response.wav", semitones=2)]

    a = augmenter.augmenter(augs)

    a.summary()


# this can only be tested visually by a human
test_augmenter_summary()

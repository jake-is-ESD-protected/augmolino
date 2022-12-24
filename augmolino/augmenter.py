from augmolino import augmentation
import numpy as np


class augmenter:

    def __init__(self, augmentations=None):
        """
        Group class which holds a dynamic amount of 
        augmentations specified by the user

        Parameters
        ----------
        `augmentations`:
            Augmentation. Single or array of augmentations from
            `augmentation.py`. If left unspecified, the augmenter
            can later be filled with augmentations via `augmenter.add()`.
            Default is `None`.

        Notes
        -----
        Depending on the `f_dest`-path of every augmentation the augmenter
        returns either an array of augmented signals from each augmentation
        or an array of `None` for each augmentation. If a location is 
        specified, the signals get stored as `.wav`-files. See "Examples"
        for more info.

        Examples
        --------
        >>> # get augmentations as signals and plot them:
        >>> from augmolino import augmenter, augmentation
        >>> import matplotlib.pyplot as plt
        >>> # specify set of augmentations:
        >>> augs = [
                augmentation.timeStretch(
                    "tests/sounds/impulse_response.wav", rate=2),
                augmentation.pitchShift(
                    "tests/sounds/impulse_response.wav", semitones=2),
                augmentation.offsetAudio(
                    "tests/sounds/impulse_response.wav", s=1)]
        >>> # create the augmenter
        >>> a = augmenter.augmenter(augs)
        >>> # run augmenter
        >>> xs = a.execute()
        >>> # create plot
        >>> fig, axs = plt.subplots(3,1)
        >>> # display signals
        >>> for i, x in enumerate(xs):
            >>> axs[i].plot(x)
        >>> plt.show()
        """

        if not augmentations:
            # init empty augmenter
            self.pipe = []

        else:
            # create array of augmentations
            if len(augmentations) > 1:
                self.pipe = augmentations
            else:
                self.pipe = [augmentations]

    def add(self, augmentation):
        """
        Add a single augmentation to the augmenter pipeline.

        Parameters
        ----------
        `augmentation`:
            Augmentation. Gets appended to existing or empty
            pipe of augmentations within the augmenter.
        """
        self.pipe.append(augmentation)

    def execute(self):
        """
        Run all augmentations within the pipe. Specific settings are
        inside of each augmentation.

        Returns
        -------
        `xs`:
            Array. Returns each augmented signal if no save location
            has been specified for the coresponding augmentation.
            Otherwise it returns an array of shape (n_augmentations, )
            filled with single `None` values.

        """
        # this is sloooow but the only way to append dynamic sizes
        xs = [[]] * len(self.pipe)
        for i, augmentation in enumerate(self.pipe):
            x = augmentation.run()

            if augmentation.f_dest == None:
                xs[i].append(x)

            else:
                xs[i].append(None)
            xs[i] = np.asarray(xs[i][i])

        return xs

    def summary(self):

        num_aug = len(self.pipe)

        print("")
        print("------------augmenter.summary------------")
        print("-----------------------------------------")
        print(f" number of augmentations: {num_aug}     ")
        print("")
        print(" type:           Source:                 ")

        for aug in self.pipe:
            print(f" > {aug.descriptor}: {aug.f_source}")

        print("------------augmenter.summary------------")
        print("")

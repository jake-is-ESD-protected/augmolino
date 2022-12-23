from augmolino import augmentation
import numpy as np


class augmenter:

    def __init__(self, augmentations=None):
        """
        Group class which holds a dynamic amount of 
        augmentations specified by the user
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
        self.pipe.append(augmentation)

    def execute(self):
        # this is sloooow but the only way to append dynamic sizes
        xs = [[]] * len(self.pipe)
        for i, augmentation in enumerate(self.pipe):
            x = augmentation.run()

            if augmentation.f_dest == None:
                xs[i].append(x)
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

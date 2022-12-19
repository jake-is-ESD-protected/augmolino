from augmolino import augmentation


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
        for augmentation in self.pipe:
            augmentation.run()

    def summary(self):

        num_aug = len(self.pipe)

        print("")
        print("------------augmenter.summary------------")
        print("-----------------------------------------")
        print(f" number of augmentations: {num_aug}")
        print("")
        print(" type:               |Source:            ")

        for aug in self.pipe:
            print(f" > {aug.descriptor}: {aug.f_source}")

        print("------------augmenter.summary------------")
        print("")

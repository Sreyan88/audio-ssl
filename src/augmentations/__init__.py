from src.augmentations.augmentations import *



class AugmentationModule:
    """The Augmentation Module"""

    def __init__(self, config):

        self.train_transform = nn.Sequential(
            self.get_augmentations(config)
        )
        if config.normalization.RunningNorm:
            self.pre_norm = RunningNorm(epoch_samples=config["normalization"]["RunningNorm"]["epoch_samples"])
        print('Augmentations:', self.train_transform)
        self.norm_status = args.use_norm

    def get_augmentations(self, config):
        list_augmentations = []

        if "MixupBYOLA" in config["Augmentations"]:
            list_augmentations.append(MixupBYOLA(ratio=config["Augmentations"]["MixupBYOLA"]["ratio"], log_mixup_exp=config["Augmentations"]["MixupBYOLA"]["log_mixup_exp"]))
        if "RandomResizeCrop" in config["Augmentations"]:
            list_augmentations.append(RandomResizeCrop(virtual_crop_scale=config["Augmentations"]["RandomResizeCrop"]["virtual_crop_scale"], freq_scale=config["Augmentations"]["RandomResizeCrop"]["freq_crop_scale"], time_scale=config["Augmentations"]["RandomResizeCrop"]["time_crop_scale"]))
        if "Kmix" in config["Augmentations"]:
            pass
        if "PatchDrop" in config["Augmentations"]:
            pass

        return list_augmentations

    def __call__(self, x):
        if self.pre_norm:
            x = self.pre_norm(x)
        # Do all models require two inputs, if not please add a condition related to a config
        return self.train_transform(x), self.train_transform(x)
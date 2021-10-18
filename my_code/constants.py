from easydict import EasyDict as edict

CONSTANTS  = edict()

CONSTANTS.env_configs = edict(
    omniglot="configs/environ/roaming-omniglot-docker.prototxt",
    rooms="configs/environ/roaming-rooms-docker.prototxt",
    tim="configs/environ/roaming-imagenet-docker.prototxt",
)

CONSTANTS.data_configs = edict(
    omniglot="configs/episodes/roaming-omniglot/roaming-omniglot-150.prototxt",
    rooms="configs/episodes/roaming-rooms/roaming-rooms-100.prototxt",
    tim="configs/episodes/roaming-imagenet/roaming-imagenet-150.prototxt",
)

CONSTANTS.datasets_stats = edict(
    tim=edict(
        mean=[120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0],
        std=[70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0],
    ),
    omniglot=edict(mean=None, std=None),
)
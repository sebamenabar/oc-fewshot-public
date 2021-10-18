import os
import sys
sys.path.insert(0, "./")

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import json
from numpy.core.numeric import NaN
from tqdm import tqdm

import numpy as np
from sklearn.metrics import pairwise

import tensorflow as tf

from constants import CONSTANTS
from fewshot.experiments.utils import get_config
from fewshot.experiments.build_model import build_pretrain_net

from fewshot.experiments.oc_fewshot import (
    ExperimentConfig,
    EpisodeConfig,
    EnvironmentConfig,
    get_data_fs,
    get_dataiter_continual,
    get_dataiter_sim,
    # build_net,
)

from dotenv import load_dotenv


def build_model(config, data_config, add_last_relu=None):
    if add_last_relu is not None:
        if config.backbone_class == "resnet_12_backbone":
            config.resnet_config.add_last_relu = add_last_relu
        else:
            config.c4_config.add_last_relu = add_last_relu

    config.num_steps = data_config.maxlen
    config.memory_net_config.max_classes = data_config.nway
    config.memory_net_config.max_stages = data_config.nstage
    config.memory_net_config.max_items = data_config.maxlen
    config.fix_unknown = data_config.fix_unknown  # Assign fix unknown ID.
    model = build_pretrain_net(config)

    return model


def build_datasets(dataset_name, data_config, env_config, nchw, seed, load_train=True):
    if dataset_name == "rooms":
        data_kwargs = {"dirpath": "./fewhost/data/matterport_split/"}
    elif dataset_name == "tim":
        data_kwargs = {"to_rgb": True}
    else:
        data_kwargs = {}

    dataset = get_data_fs(env_config, load_train=load_train, **data_kwargs)
    if dataset_name == "rooms":
        data = get_dataiter_sim(
            dataset,
            data_config,
            batch_size=1,
            nchw=nchw,
            seed=seed,
        )
    else:
        data = get_dataiter_continual(
            dataset,
            data_config,
            batch_size=1,
            # nchw=model.config.data_format == "NCHW",
            nchw=nchw,
            save_additional_info=True,
            random_box=data_config.random_box,
            seed=seed,
            mean=CONSTANTS.datasets_stats[dataset_name].mean,
            std=CONSTANTS.datasets_stats[dataset_name].std,
        )

    return dataset, data


def main(args):
    if args.data_config_fp is None:
        args.data_config_fp = CONSTANTS.data_configs[args.dataset]
    if args.env_config_fp is None:
        args.env_config_fp = CONSTANTS.env_configs[args.dataset]
    config = get_config(args.model_config_fp, ExperimentConfig)
    data_config = get_config(args.data_config_fp, EpisodeConfig)
    env_config = get_config(args.env_config_fp, EnvironmentConfig)

    env_config.results = "./results/oc-fewshot/"
    if args.dataset == "omniglot":
        env_config.data_folder = "./data/roaming-omniglot/"

    model = build_model(config, data_config, args.add_last_relu)
    model.load(args.model_dir)
    if args.nway is not None:
        data_config.nway = args.nway
    if args.maxlen is not None:
        data_config.maxlen = args.maxlen
    datasets, data = build_datasets(
        args.dataset,
        data_config,
        env_config,
        model.backbone._config.data_format == "NCHW",
        args.seed,
    )


    if args.num_test_steps == -1:
        args.num_test_steps = len(data[args.split])
    nsteps = args.num_test_steps
    pbar = tqdm(zip(range(0, nsteps), data[args.split]), total=nsteps)
    nns = []
    for i, batch in pbar:
        x = batch['x_s']
        y = batch['y_s']
        x = tf.squeeze(x, 0)
        _y = tf.squeeze(y, 0)
        feats = model.backbone(x, is_training=False)
        c_distances = pairwise.cosine_distances((feats))
        nn = (_y.numpy() == _y.numpy()[(c_distances + 2 * np.eye(c_distances.shape[0])).argmin(axis=-1)]).mean()
        nns.append(nn)

        if (i % 10) == 0:
            pbar.set_postfix(nn=np.mean(nns) * 100)
    print("\nFinal")
    print(np.mean(nns))


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", default=0)
    parser.add_argument("--num_test_steps", type=int, default=-1)
    parser.add_argument("--split", choices=["trainval_fs", "val_fs", "test_fs"], default="val_fs")
    parser.add_argument("--nway", type=int)
    parser.add_argument("--maxlen", type=int)

    parser.add_argument(
        "--dataset", choices=["omniglot", "rooms", "tim"], required=True
    )
    parser.add_argument("--data_config_fp")
    parser.add_argument("--env_config_fp")
    parser.add_argument("--model_config_fp")

    parser.add_argument("--model_dir")
    # parser.add_argument("--model_fname")

    parser.add_argument("--add_last_relu")
    # parser.add_argument("--adapt")

    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    main(parse_args())

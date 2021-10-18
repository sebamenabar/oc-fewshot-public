import os
import sys

sys.path.insert(0, "./")

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import json
from tqdm import tqdm

from comet_ml import Experiment
import numpy as np
from sklearn.metrics import pairwise


import tensorflow as tf

import logging
from logging import handlers

from constants import CONSTANTS

from fewshot.experiments.oc_fewshot import evaluate, get_stats
from fewshot.experiments.utils import get_config
from fewshot.experiments.build_model import build_net, build_pretrain_net

from fewshot.experiments.oc_fewshot import (
    ExperimentConfig,
    EpisodeConfig,
    EnvironmentConfig,
    get_data_fs,
    get_dataiter_continual,
    get_dataiter_sim,
    # build_net,
)
from fewshot.experiments.get_data_iter import get_dataiter
from fewshot.models.nets.net import Net, ContainerModule
from fewshot.models.modules.nnlib import Linear
from my_code.finetune import get_loaders, build_datasets, init_writers, evaluate_many

from dotenv import load_dotenv

def build_model(config, data_config, add_last_relu=None, num_classes=None):
    if add_last_relu is not None:
        if config.backbone_class == "resnet_12_backbone":
            config.resnet_config.add_last_relu = add_last_relu
        else:
            config.c4_config.add_last_relu = add_last_relu

    # config.memory_net_config.radius_init = 10
    # config.memory_net_config.radius_init_write = 10

    if num_classes is not None:
        config.num_classes = num_classes

    config.num_steps = data_config.maxlen
    config.memory_net_config.max_classes = data_config.nway
    config.memory_net_config.max_stages = data_config.nstage
    config.memory_net_config.max_items = data_config.maxlen
    config.fix_unknown = data_config.fix_unknown  # Assign fix unknown ID.
    model = build_pretrain_net(config)

    return model

def main(args):
    logger, comet, writer, logdir = init_writers(
        args.logdir,
        args.exp_name,
        args.logcomet,
        # args.comet_api_key,
        args.comet_project_name,
        args.seed,
        args.exist_ok,
        args,
    )

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


    if config.backbone_class == "resnet_12_backbone":
        bb_config = config.resnet_config
    else:
        bb_config = config.c4_config

    datasets, data = build_datasets(
        # args.batch_size,
        1,
        args.dataset,
        data_config,
        env_config,
        bb_config.data_format == "NCHW",
        args.seed,
    )

    if args.dataset == "tim":
        num_classes = 351
    elif args.dataset == "rooms":
        num_classes = None
    elif args.dataset == "omniglot":
        num_classes = len(datasets["train_fs"].cls_dict)

    model = build_model(config, data_config, args.add_last_relu, num_classes=num_classes)
    # mem_model = build_net(
    #     config,
    #     backbone=model.backbone,
    #     # learn_temp=args.learn_temp,
    # )


    dataset_name = args.dataset
    if args.use_dataset_stats:
        mean=CONSTANTS.datasets_stats[dataset_name].mean
        std=CONSTANTS.datasets_stats[dataset_name].std
    else:
        mean = std = None
    # mean = std = None

    data = get_dataiter(
    #   dataset,
        {k.replace("_fs", ""): v for k, v in datasets.items()},
        # config.optimizer_config.batch_size,
        args.batch_size,
        nchw=model.backbone.config.data_format == 'NCHW',
        data_aug=True,
        random_box=data_config.random_box,
        # seed=args.seed,
        mean=mean,
        std=std,
        da_prep2=args.da_prep2,
        min_object_covered=args.min_object_covered,
    )

    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        args.lr_decay_steps, args.lr_list,
    )
    optimizer = tf.optimizers.SGD(
        learning_rate=learning_rate_fn,
        momentum=0.9,
    )

    loss_fn = lambda y, outputs: tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=y)
    )

    @tf.function
    def train_step(x, y):
        with tf.GradientTape(persistent=True) as tape:
            outputs = model(x)
            _loss = loss_fn(y, outputs)
            _wd = args.weight_decay * tf.add_n(
                [tf.reduce_sum(w ** 2) * (1 / 2) for w in model.regularized_weights()]
            )
            loss = _loss + _wd

        gradients = tape.gradient(loss, model.var_to_optimize())
        optimizer.apply_gradients(zip(gradients, model.var_to_optimize()))

        # if args.loss == "sup":
        acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(outputs, -1), y), tf.float32)
        )
        # else:
        #     acc = 0.0

        return acc, _loss, _wd

    acc = None
    loss = None
    alpha = 0.99

    num_steps = args.num_steps
    pbar = tqdm(range(num_steps), total=num_steps, ncols=0, mininterval=3)
    i = -1
    train_loader = data["train"].get_dataset()
    while i < (num_steps - 1):
        for b in train_loader:
            x = b["x"]
            y= b["y"]
            i += 1
            if i >= num_steps:
                break
            _acc, _loss, wd = train_step(x, tf.cast(y, tf.int64))

            if acc is None:
                acc = _acc
                loss = _loss
            else:
                acc = alpha * acc + (1 - alpha) * _acc
                loss = alpha * loss + (1 - alpha) * _loss

            if (i % 50) == 0:
                _acc = acc.numpy()
                _loss = loss.numpy()
                _wd = wd.numpy()

                pbar.set_postfix(
                    acc=_acc,
                    loss=_loss,
                    wd=_wd,
                )

                tf.summary.scalar("train/acc_ema", _acc, step=i + 1)
                tf.summary.scalar("train/loss_ema", _loss, step=i + 1)
                tf.summary.scalar("train/wd_ema", _wd, step=i + 1)

            if (i == 0) or (((i + 1) % 5000) == 0) or ((i + 1) == num_steps):
                logger.info("")
                logger.info(f"Step {i}")
                logger.info(f"Loss {loss.numpy()}")
                logger.info(f"Acc {acc.numpy()}")
                logger.info(f"Wd {wd.numpy()}")

            if (i + 1) in args.lr_decay_steps:
                model.save(f"{logdir}/model-{i}.pkl")
                # save_optimizer(f"{logdir}/optimizer-{i}.pkl", optimizer)
                # if args.num_layers_mlp > 0:
                #     save_mlp(f"{logdir}/mlp-{i}.pkl", mlp)

            if (i % 5000) == 0:
                model.save(f"{logdir}/model-curr.pkl")
                # save_optimizer(f"{logdir}/optimizer-curr.pkl", optimizer)
                # if args.num_layers_mlp > 0:
                #     save_mlp(f"{logdir}/mlp-curr.pkl", mlp)

            pbar.update()

    model.save(f"{logdir}/model-last.pkl")
    # save_optimizer(f"{logdir}/optimizer-last.pkl", optimizer)
    # if args.num_layers_mlp > 0:
    #     save_mlp(f"{logdir}/mlp-last.pkl", mlp)



def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", default=0, type=int)
    # parser.add_argument("--num_test_steps", type=int, default=-1)
    # parser.add_argument(
    #     "--split", choices=["trainval_fs", "val_fs", "test_fs"], default="val_fs"
    # )
    # parser.add_argument("--nway", type=int)
    # parser.add_argument("--maxlen", type=int)

    parser.add_argument("--use_dataset_stats", default=False, action="store_true")
    parser.add_argument("--da_prep2", default=False, action="store_true")
    parser.add_argument("--min_object_covered", default=0.2, type=float)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_list", type=float, nargs="+", default=[5e-2, 5e-3, 5e-4])
    parser.add_argument("--lr_decay_steps", type=int, nargs="+", default=[100000, 150000])
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_steps", type=int, default=200000)
    parser.add_argument("--num_train_test_steps", type=int, default=500)
    parser.add_argument("--num_test_test_steps", type=int, default=2000)
    parser.add_argument("--test_interval", type=int, default=100000)

    parser.add_argument(
        "--dataset", choices=["omniglot", "rooms", "tim"], required=True
    )
    parser.add_argument("--data_config_fp")
    parser.add_argument("--env_config_fp")
    parser.add_argument("--model_config_fp")

    parser.add_argument("--model_dir")
    # parser.add_argument("--model_fname")

    parser.add_argument("--exist_ok", action="store_true", default=False)
    parser.add_argument("--logdir")
    parser.add_argument("--exp_name")

    parser.add_argument("--learn_temp", default=False, action="store_true")
    parser.add_argument("--add_last_relu", type=int, choices=[0, 1])
    parser.add_argument("--hidden_dim", type=int)
    # parser.add_argument("--adapt", default=False, action="store_true")

    parser.add_argument("--logcomet", action="store_true", default=False)
    parser.add_argument("--comet_project_name", default="wandering")
    # parser.add_argument("--comet_api_key", default="wandering")

    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    main(parse_args())

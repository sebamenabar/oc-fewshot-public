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
from fewshot.models.nets.net import Net, ContainerModule
from fewshot.models.modules.nnlib import Linear

from dotenv import load_dotenv


class Adapt(ContainerModule):
    def __init__(self, last_relu, backbone, in_size, hidden_dim, dtype=tf.float32):
        super().__init__(dtype=dtype)
        self.last_relu = last_relu
        self.backbone = backbone
        self.in_size = in_size
        self.hidden_dim = hidden_dim

        print(in_size)
        print(hidden_dim)

        self.linear1 = Linear("adapt1", in_size, hidden_dim)

    def forward(self, *args, **kwargs):
        out = self.backbone(*args, **kwargs)
        out = self.linear1(out)
        if self.last_relu:
            out = tf.nn.relu(out)
        return out

    def get_output_dimension(self, *args, **kwargs):
        return self.backbone.get_output_dimension(*args, **kwargs)
        # return self.out_size

    @property
    def config(self):
        return self.backbone.config


def build_model(config, data_config, add_last_relu=None):
    if add_last_relu is not None:
        if config.backbone_class == "resnet_12_backbone":
            config.resnet_config.add_last_relu = add_last_relu
        else:
            config.c4_config.add_last_relu = add_last_relu

    # config.memory_net_config.radius_init = 10
    # config.memory_net_config.radius_init_write = 10

    config.num_steps = data_config.maxlen
    config.memory_net_config.max_classes = data_config.nway
    config.memory_net_config.max_stages = data_config.nstage
    config.memory_net_config.max_items = data_config.maxlen
    config.fix_unknown = data_config.fix_unknown  # Assign fix unknown ID.
    model = build_pretrain_net(config)

    return model


def get_loaders(dataset_name, dataset, data_config, batch_size, nchw, seed):
    if dataset_name == "rooms":
        data = get_dataiter_sim(
            dataset,
            data_config,
            batch_size=batch_size,
            nchw=nchw,
            seed=seed,
        )
    else:
        data = get_dataiter_continual(
            dataset,
            data_config,
            batch_size=batch_size,
            # nchw=model.config.data_format == "NCHW",
            nchw=nchw,
            save_additional_info=True,
            random_box=data_config.random_box,
            seed=seed,
            mean=CONSTANTS.datasets_stats[dataset_name].mean,
            std=CONSTANTS.datasets_stats[dataset_name].std,
        )
    return data


def build_datasets(
    batch_size, dataset_name, data_config, env_config, nchw, seed, load_train=True
):
    if dataset_name == "rooms":
        data_kwargs = {"dirpath": "./fewhost/data/matterport_split/"}
    elif dataset_name == "tim":
        data_kwargs = {"to_rgb": True}
    else:
        data_kwargs = {}

    dataset = get_data_fs(env_config, load_train=load_train, **data_kwargs)
    data = get_loaders(dataset_name, dataset, data_config, batch_size, nchw, seed)

    return dataset, data


def init_writers(logdir, exp_name, logcomet, comet_project_name, seed, exist_ok, args):
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    exp_name = f"{exp_name}_{seed}"
    logdir = os.path.join(logdir, exp_name)
    if logdir:
        os.makedirs(logdir, exist_ok=bool(exist_ok))
        with open(os.path.join(logdir, "args.json"), "w") as f:
            json.dump(vars(args), f)

    fh = logging.FileHandler(os.path.join(logdir, "log.txt"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        # logging.Formatter('rank:' + str(args['rank']) + ' ' + name + ' %(levelname)-8s %(message)s'))
        logging.Formatter("%(levelname)-8s %(message)s")
    )

    logger = logging.getLogger("experiment")
    logger.addHandler(fh)

    ch = logging.handlers.logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(
        # logging.Formatter('rank:' + str(args['rank']) + ' ' + name + ' %(levelname)-8s %(message)s'))
        logging.Formatter("%(levelname)-8s %(message)s")
    )
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    logger.info(str(vars(args)))

    writer = tf.summary.create_file_writer(os.path.join(logdir, "tensorboard"))
    writer.set_as_default()

    if logcomet:
        comet_api_key = os.environ["COMET_API_KEY"]
    else:
        comet_api_key = os.environ.get("COMET_API_KEY", default="")
    comet = Experiment(
        api_key=comet_api_key,
        workspace="paper-cl-reps",
        project_name=comet_project_name,
        disabled=not logcomet,
    )
    comet.set_name(exp_name)
    comet.log_parameters(vars(args))

    return logger, comet, writer, logdir


def log_metrics(logger, comet, stats, nns, step, prefix):
    logger.info(f"Stats step {step} {prefix}")
    # print(nns)
    nns = np.array(nns) * 100
    # print(nns)
    nns_mean = np.mean(nns)
    nns_std = np.std(nns)
    logger.info(f"NNs {nns_mean}±{nns_std}")
    logger.info(f"AP: {str(stats['ap'] * 100)}")
    logger.info(f"n-shot: {str(stats['acc_nshot'] * 100)}")
    logger.info(f"n-shot se: {str(stats['acc_nshot_se'] * 100)}")
    # logger.info("")

    nshots_str = ";".join([
        f"{ns:.3f}±{se:.3f}" for ns, se in zip(stats["acc_nshot"] * 100, stats["acc_nshot_se"] * 100)
    ])
    logger.info(f"{stats['ap'] * 100:.3f};{nshots_str};{nns_mean:.3f}±{nns_std:.3f}")

    tf.summary.scalar(f"{prefix}/ap", stats["ap"], step=0)
    tf.summary.scalar(f"{prefix}/acc_nshot1", stats["acc_nshot"][0], step=step)
    tf.summary.scalar(f"{prefix}/acc_nshot2", stats["acc_nshot"][1], step=step)
    tf.summary.scalar(f"{prefix}/acc_nshot3", stats["acc_nshot"][2], step=step)
    tf.summary.scalar(f"{prefix}/acc_nshot4", stats["acc_nshot"][3], step=step)
    tf.summary.scalar(f"{prefix}/acc_nshot5", stats["acc_nshot"][4], step=step)
    tf.summary.scalar(f"{prefix}/nn", nns_mean, step=step)
    tf.summary.scalar(f"{prefix}/nn_std", nns_std, step=step)

    comet.log_metrics(
        {
            "ap": stats["ap"],
            "acc_nshot1": stats["acc_nshot"][0],
            "acc_nshot2": stats["acc_nshot"][1],
            "acc_nshot3": stats["acc_nshot"][2],
            "acc_nshot4": stats["acc_nshot"][3],
            "acc_nshot5": stats["acc_nshot"][4],
            "nns": nns_mean,
            "nns_std": nns_std,
        },
        prefix=prefix,
        step=step,
    )


def evaluate_many(
    model,
    loaders,
    keys,
    logger,
    comet,
    step,
    maxlen,
    num_steps,
):
    logger.info("")
    logger.info(f"Evaluating")
    out = {}
    for k, ns in zip(keys, num_steps):
        logger.info(f"Evaluating {k}")
        loaders[k].reset()
        r1, nns = evaluate(model, loaders[k], ns, verbose=True)
        # nns = np.mean(nns)
        stats = get_stats(r1, tmax=maxlen)
        logger.info(f"")
        # logger.info(f"Stats {k}")
        # logger.info(str(stats))
        log_metrics(logger, comet, stats, nns, step, k)
        logger.info("")
        out[k] = (stats, np.mean(nns))

    return out


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

    if args.lr_list is not None:
        config.optimizer_config.lr_list[:] = args.lr_list
    if args.lr_decay_steps is not None:
        config.optimizer_config.lr_decay_steps[:] = args.lr_decay_steps

    model = build_model(config, data_config, args.add_last_relu)
    if args.hidden_dim is None:
        hidden_dim = model.backbone.get_output_dimension()[0]
    else:
        hidden_dim = args.hidden_dim
    model.set_trainable(False)
    if args.adapt:
        backbone = Adapt(
            args.add_last_relu,
            model.backbone,
            model.backbone.get_output_dimension()[0],
            hidden_dim,
        )
    else:
        backbone = model.backbone

    mem_model = build_net(
        config,
        backbone=backbone,
        learn_temp=args.learn_temp,
    )
    mem_model.load(args.model_dir)

    for w in mem_model.var_to_optimize():
        logger.info(f"{w.name}, {w.shape}")

    if args.nway is not None:
        data_config.nway = args.nway
    if args.maxlen is not None:
        data_config.maxlen = args.maxlen
    datasets, data = build_datasets(
        # args.batch_size,
        config.optimizer_config.batch_size,
        args.dataset,
        data_config,
        env_config,
        model.backbone._config.data_format == "NCHW",
        args.seed,
    )

    all_stats = evaluate_many(
        mem_model,
        data,
        ["trainval_fs", "val_fs", "test_fs"],
        logger,
        comet,
        0,
        data_config.maxlen,
        [args.num_train_test_steps, args.num_test_test_steps, args.num_test_test_steps],
    )

    train_ap = all_stats["trainval_fs"][0]["ap"]
    best_val_ap = val_ap = all_stats["val_fs"][0]["ap"]
    test_ap = all_stats["test_fs"][0]["ap"]

    train_nns = all_stats["trainval_fs"][1]
    best_val_nns = val_nns = all_stats["val_fs"][1]
    test_nns = all_stats["test_fs"][1]

    # best_val_ap = val_ap
    # best_val_nns = all_stats["val_fs"][1]
    tf.summary.scalar("val/best_ap", best_val_ap, step=0)
    tf.summary.scalar("val/best_nns", best_val_nns, step=0)
    tf.summary.scalar("test/best_ap", test_ap, step=0)
    tf.summary.scalar("test/best_nns", test_nns, step=0)
    comet.log_metrics(
        {
            "best_ap": best_val_ap,
            "best_nns": best_val_nns,
            },
        prefix="val",
        step=0,
    )
    comet.log_metrics(
        {
            "best_ap": test_ap,
            "best_nns": test_nns,
            },
        prefix="test",
        step=0,
    )

    pbar = tqdm(zip(range(0, args.num_steps), data["train_fs"]), total=args.num_steps, ncols=0)
    loss_ema = None
    wd_ema = None
    for i, batch in pbar:
        x = batch["x_s"]
        y = batch["y_s"]
        y = batch["y_s"]
        y_gt = batch["y_gt"]

        kwargs = {
            "y_gt": y_gt,
            "flag": batch["flag_s"],
            "backbone_is_training": False,
            # "last_is_training": last_is_training,
            "return_metric": True,
        }
        # kwargs['writer'] = writer

        loss, metric = mem_model.train_step(x, y, **kwargs)

        if np.isnan(loss):
            comet.log_metric("nan", 1, step=i + 1)
            logger.info("")
            logger.info("NAN, ending")
            # comet.end()
            # sys.exit()

        wd = metric["wd"]
        ema_alpha = 0.95
        if loss_ema is None:
            loss_ema = loss
            wd_ema = wd
        else:
            loss_ema = ema_alpha * loss_ema + (1 - ema_alpha) * loss
            wd_ema = ema_alpha * wd_ema + (1 - ema_alpha) * wd

        if ((i % 25) == 0) or ((i + 1) == args.num_steps):
            pbar.set_postfix(
                loss=loss.numpy(),
                loss_ema=loss_ema.numpy(),
                wd=wd.numpy(),
                wd_ema=wd_ema.numpy(),
                v_ap=f"{val_ap:.4f}",
                b_ap=f"{best_val_ap:.4f}",
                tr_ap=f"{train_ap:.4f}",
                te_ap=f"{test_ap:.4f}",
                v_nns=f"{val_nns:.4f}",
                b_nns=f"{best_val_nns:.4f}",
                tr_nns=f"{train_nns:.4f}",
                te_nns=f"{test_nns:.4f}",
                # lr=mem_model.memory.lr.numpy(),
                # temp=mem_model.memory._temp.numpy(),
            )
            tf.summary.scalar("wd", wd, step=i + 1)
            tf.summary.scalar("wd_ema", wd_ema, step=i + 1)
            tf.summary.scalar("train/loss", loss, step=i + 1)
            tf.summary.scalar("train/loss_ema", loss_ema, step=i + 1)
            comet.log_metrics(
                {
                    "loss": loss.numpy(),
                    "loss_ema": loss_ema.numpy(),
                    "wd": wd.numpy(),
                    "wd_ema": wd_ema.numpy(),
                },
                prefix="train",
                step=i + 1,
            )

        if (i != 0) and (
            ((i % args.test_interval) == 0) or ((i + 1) == args.num_steps)
        ):
            # if args.num_test_test_steps == -1:
            if True:
                _data = get_loaders(
                    args.dataset,
                    datasets,
                    data_config,
                    args.batch_size,
                    model.backbone._config.data_format == "NCHW",
                    args.seed,
                )
            else:
                _data = data

            logger.info("")
            all_stats = evaluate_many(
                mem_model,
                _data,
                ["trainval_fs", "val_fs", "test_fs"],
                logger,
                comet,
                i + 1,
                data_config.maxlen,
                [
                    args.num_train_test_steps,
                    args.num_test_test_steps,
                    args.num_test_test_steps,
                ],
            )

            val_ap = all_stats["val_fs"][0]["ap"]
            test_ap = all_stats["test_fs"][0]["ap"]
            test_nns = all_stats["test_fs"][1]
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_val_nns = all_stats["val_fs"][1]
                tf.summary.scalar("val/best_ap", best_val_ap, step=i + 1)
                tf.summary.scalar("val/best_nns", best_val_nns, step=i + 1)
                tf.summary.scalar("test/best_ap", test_ap, step=i + 1)
                tf.summary.scalar("test/best_nns", test_nns, step=i + 1)
                comet.log_metrics(
                    {"best_ap": best_val_ap, "best_nns": best_val_nns},
                    prefix="val",
                    step=i + 1,
                )
                comet.log_metrics(
                    {"best_ap": test_ap, "best_nns": test_nns},
                    prefix="test",
                    step=i + 1,
                )
                logger.info(" -------- BEST VAL AP --------")
                logger.info(" -------- BEST VAL AP --------")

            train_ap = all_stats["trainval_fs"][0]["ap"]
            val_ap = all_stats["val_fs"][0]["ap"]
            test_ap = all_stats["test_fs"][0]["ap"]

            train_nns = all_stats["trainval_fs"][1]
            val_nns = all_stats["val_fs"][1]
            test_nns = all_stats["test_fs"][1]

            pbar.set_postfix(
                loss=loss.numpy(),
                loss_ema=loss_ema.numpy(),
                wd=wd.numpy(),
                wd_ema=wd_ema.numpy(),
                v_ap=f"{val_ap:.4f}",
                b_ap=f"{best_val_ap:.4f}",
                tr_ap=f"{train_ap:.4f}",
                te_ap=f"{test_ap:.4f}",
                v_nns=f"{val_nns:.4f}",
                b_nns=f"{best_val_nns:.4f}",
                tr_nns=f"{train_nns:.4f}",
                te_nns=f"{test_nns:.4f}",
                # lr=mem_model.memory.lr.numpy(),
                # temp=mem_model.memory._temp.numpy(),
            )

            mem_model.save(f"{logdir}/model-best.pkl")
    mem_model.save(f"{logdir}/model-last.pkl")


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", default=0, type=int)
    # parser.add_argument("--num_test_steps", type=int, default=-1)
    parser.add_argument(
        "--split", choices=["trainval_fs", "val_fs", "test_fs"], default="val_fs"
    )
    parser.add_argument("--nway", type=int)
    parser.add_argument("--maxlen", type=int)

    # parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_list", type=float, nargs="+")
    parser.add_argument("--lr_decay_steps", type=int, nargs="+", default=[])
    parser.add_argument("--num_steps", type=int, default=6000)
    parser.add_argument("--num_train_test_steps", type=int, default=500)
    parser.add_argument("--num_test_test_steps", type=int, default=2000)
    parser.add_argument("--test_interval", type=int, default=1000)

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
    parser.add_argument("--adapt", default=False, action="store_true")

    parser.add_argument("--logcomet", action="store_true", default=False)
    parser.add_argument("--comet_project_name", default="wandering")
    # parser.add_argument("--comet_api_key", default="wandering")

    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    main(parse_args())

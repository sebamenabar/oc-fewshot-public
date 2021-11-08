"""
Train an online few-shot network.
Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle as pkl
import six
import tensorflow as tf
from sklearn.metrics import pairwise
import numpy as np

from tqdm import tqdm

from fewshot.configs.environment_config_pb2 import EnvironmentConfig
from fewshot.configs.episode_config_pb2 import EpisodeConfig
from fewshot.configs.experiment_config_pb2 import ExperimentConfig
from fewshot.experiments.build_model import build_net
from fewshot.experiments.build_model import build_pretrain_net
from fewshot.experiments.get_data_iter import get_dataiter_continual
from fewshot.experiments.get_data_iter import get_dataiter_sim
from fewshot.experiments.get_stats import get_stats
from fewshot.experiments.get_stats import log_results
from fewshot.experiments.utils import ExperimentLogger
from fewshot.experiments.utils import get_config
from fewshot.experiments.utils import get_data_fs
from fewshot.experiments.utils import latest_file
from fewshot.experiments.utils import save_config
from fewshot.utils.logger import get as get_logger

log = get_logger()


class dummy_context_mgr():

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False


def evaluate(model, dataiter, num_steps, verbose=False):
  """Evaluates online few-shot episodes.
  Args:
    model: Model instance.
    dataiter: Dataset iterator.
    num_steps: Number of episodes.
  """
  if num_steps == -1:
    it = six.moves.xrange(len(dataiter))
  else:
    it = six.moves.xrange(num_steps)
  if verbose:
    it = tqdm(it, ncols=0)
  results = []
  nns = []
  for i, batch in zip(it, dataiter):

    # print(batch)

    x = batch['x_s']
    y = batch['y_s']
    y_gt = batch['y_gt']
    y_full = batch['y_full']
    train_flag = batch['flag_s']
    kwargs = {}

    if train_flag is not None:
      flag = train_flag.numpy()
    else:
      flag = None
    pred = model.eval_step(x, y, **kwargs)

    x = tf.squeeze(x, 0)
    _y = tf.squeeze(y, 0)
    feats = model.backbone(x, is_training=False)

    c_distances = pairwise.cosine_distances((feats))
    nn = (_y.numpy() == _y.numpy()[(c_distances + 2 * np.eye(c_distances.shape[0])).argmin(axis=-1)]).mean()

    # Support set metrics, accumulate per number of shots.
    y_np = y_full.numpy()  # [B, T]
    y_s_np = y.numpy()  # [B, T]
    y_gt_np = y_gt.numpy()  # [B, T]
    pred_np = pred.numpy()  # [B, T, K]
    pred_id_np = model.predict_id(pred).numpy()  # [B, T]

    # print('-' * 50)
    # import numpy as np
    # print('y_gt pred unk', np.stack([y_gt_np, pred_id_np,
    #                                  pred_np[:, :, -1]]).T)
    # # print('pred', pred_id_np)
    # print('-' * 50)

    results.append({
        'y_full': y_np,
        'y_gt': y_gt_np,
        'y_s': y_s_np,
        'pred': pred_np,
        'pred_id': pred_id_np,
        'flag': flag,
        # 'nn': nn,
    })
    nns.append(nn)
  return results, nns


def train(model,
          dataiter,
          dataiter_traintest,
          dataiter_test,
          dataiter_test_test,
          ckpt_folder,
          final_save_folder=None,
          nshot_max=5,
          maxlen=40,
          logger=None,
          writer=None,
          in_stage=False,
          is_chief=True,
          reload_flag=None,
          backbone_is_training=True):
  """Trains the online few-shot model.
  Args:
    model: Model instance.
    dataiter: Dataset iterator.
    dataiter_test: Dataset iterator for validation.
    save_folder: Path to save the checkpoints.
  """
  N = model.max_train_steps
  config = model.config.train_config

  def try_log(*args, **kwargs):
    if logger is not None:
      logger.log(*args, **kwargs)

  def try_flush(*args, **kwargs):
    if logger is not None:
      logger.flush()

  r = None
  keep = True
  restart = False
  best_val = 0.0
  results = None
  while keep:
    keep = False
    start = model.step.numpy()
    if start > 0:
      log.info('Restore from step {}'.format(start))
    it = six.moves.xrange(start, N)
    if is_chief:
      it = tqdm(it, ncols=0)
    for i, batch in zip(it, dataiter):

      # print(batch)

      tf.summary.experimental.set_step(i + 1)
      x = batch['x_s']
      y = batch['y_s']
      y = batch['y_s']
      y_gt = batch['y_gt']

      kwargs = {'y_gt': y_gt, 'flag': batch['flag_s'], "backbone_is_training": backbone_is_training}
      kwargs['writer'] = writer

      loss = model.train_step(x, y, **kwargs)

      if i == start and reload_flag is not None and not restart:
        print(reload_flag)
        model.load(reload_flag, load_optimizer=True)

      # Synchronize distributed weights.
      if i == start and model._distributed and not restart:
        import horovod.tensorflow as hvd
        hvd.broadcast_variables(model.var_to_optimize(), root_rank=0)
        hvd.broadcast_variables(model.optimizer.variables(), root_rank=0)
        if model.config.set_backbone_lr:
          hvd.broadcast_variables(model._bb_optimizer.variables(), root_rank=0)

      if loss > 10.0 and i > start + config.steps_per_save:
        # Something wrong happened.
        log.error('Something wrong happened. loss = {}'.format(loss))
        import pickle as pkl
        pkl.dump(batch, open(
            os.path.join(final_save_folder, 'debug.pkl'), 'wb'))
        log.error('debug file dumped')
        restart = True
        keep = True

        latest = latest_file(ckpt_folder, 'weights-')
        model.load(latest_file)
        log.error('Reloaded latest checkpoint from {}'.format(latest_file))
        break

      # Evaluate.
      
      if is_chief and ((i + 1) % config.steps_per_val == 0 or i == 0):
        results = {}
        for key, data_it_ in zip(['train', 'val', 'test'],
                                 [dataiter_traintest, dataiter_test, dataiter_test_test]):
          data_it_.reset()
          r1, nns = evaluate(model, data_it_, 120)
          # print(r1)
          # print(nns)
          r = get_stats(r1, nshot_max=nshot_max, tmax=maxlen)
          nn = np.mean(nns)
          for s in range(nshot_max):
            try_log('online fs acc {}/s{}'.format(key, s), i + 1,
                    r['acc_nshot'][s] * 100.0)
          try_log('online fs ap {}'.format(key), i + 1, r['ap'] * 100.0)
          try_log('nn {}'.format(key), i + 1, nn * 100.0)

          results[key] = (r, nn)

        try_log('lr', i + 1, model.learn_rate())
        print()

      # Save.
      if results is not None:
        r = results['val'][0]
      if is_chief and ((i + 1) % config.steps_per_save == 0 or i == 0):
        model.save(os.path.join(ckpt_folder, 'weights-{}'.format(i + 1)))

        # Save the best checkpoint.
        if r is not None:
          if r['ap'] > best_val:
            model.save(
                os.path.join(final_save_folder, 'best-{}'.format(i + 1)))
            best_val = r['ap']

      # Write logs.
      if is_chief and ((i + 1) % config.steps_per_log == 0 or i == 0):
        try_log('loss', i + 1, loss)
        try_flush()

        # Update progress bar.
        post_fix_dict = {}
        post_fix_dict['lr'] = '{:.3e}'.format(model.learn_rate())
        post_fix_dict['loss'] = '{:.3e}'.format(loss)
        if r is not None:
          post_fix_dict['ap_val'] = '{:.3f}'.format(results["val"][0]['ap'] * 100.0)
          post_fix_dict['ap_test'] = '{:.3f}'.format(results["test"][0]['ap'] * 100.0)
          post_fix_dict['nn_val'] = '{:.3f}'.format(results["val"][1] * 100.0)
          post_fix_dict['nn_train'] = '{:.3f}'.format(results["train"][1] * 100.0)
          post_fix_dict['nn_test'] = '{:.3f}'.format(results["test"][1] * 100.0)
        it.set_postfix(**post_fix_dict)

  # Save.
  if is_chief and final_save_folder is not None:
    model.save(os.path.join(final_save_folder, 'weights-{}'.format(N)))


def main():
  assert tf.executing_eagerly(), 'Only eager mode is supported.'
  assert args.config is not None, 'You need to pass in model config file path'
  assert args.data is not None, 'You need to pass in episode config file path'
  assert args.env is not None, 'You need to pass in environ config file path'
  assert args.tag is not None, 'You need to specify a tag'
  global log


  
  config = get_config(args.config, ExperimentConfig)
  data_config = get_config(args.data, EpisodeConfig)
  env_config = get_config(args.env, EnvironmentConfig)


  save_folder = os.path.join(env_config.results, env_config.dataset, args.tag)
  # Create save folder.
  if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

  if args.file_logger:
    log = get_logger(os.path.join(save_folder, f"log.txt"))

  log.info('Command line args {}'.format(args))
  log.info('Model: \n{}'.format(config))
  log.info('Data episode: \n{}'.format(data_config))
  log.info('Environment: \n{}'.format(env_config))
  log.info('Number of classes {}'.format(data_config.nway))


  config.num_classes = data_config.nway  # Assign num classes.
  config.num_steps = data_config.maxlen
  config.memory_net_config.max_classes = data_config.nway
  config.memory_net_config.max_stages = data_config.nstage
  config.memory_net_config.max_items = data_config.maxlen
  config.oml_config.num_classes = data_config.nway
  config.fix_unknown = data_config.fix_unknown  # Assign fix unknown ID.

  if 'SLURM_JOB_ID' in os.environ:
    log.info('SLURM job ID: {}'.format(os.environ['SLURM_JOB_ID']))


  if not args.reeval:
    model = build_pretrain_net(config)
    # mem_model = build_net(config, backbone=model.backbone, learn_temp=args.learn_temp)
    mem_model = build_net(config, backbone=model.backbone)
    reload_flag = None
    restore_steps = 0
    # Checkpoint folder.
    ckpt_path = env_config.checkpoint
    if len(ckpt_path) > 0 and os.path.exists(ckpt_path):
      ckpt_folder = os.path.join(ckpt_path, os.environ['SLURM_JOB_ID'])
    else:
      ckpt_folder = save_folder

    # Reload previous checkpoint.
    # if os.path.exists(ckpt_folder) and not args.eval:
    #   latest = latest_file(ckpt_folder, 'weights-')
    #   if latest is not None:
    #     log.info('Checkpoint already exists. Loading from {}'.format(latest))
    #     mem_model.load(latest)  # Not loading optimizer weights here.
    #     reload_flag = latest
    #     restore_steps = int(reload_flag.split('-')[-1])
    if args.resume:
      latest = latest_file(ckpt_folder, 'weights-')
      if latest is not None:
        log.info('Checkpoint already exists. Loading from {}'.format(latest))
        mem_model.load(latest)  # Not loading optimizer weights here.
        reload_flag = latest
        restore_steps = int(reload_flag.split('-')[-1])


    if not args.eval:
      save_config(config, save_folder)

    # Create TB logger.
    if not args.eval:
      writer = tf.summary.create_file_writer(save_folder)
      logger = ExperimentLogger(writer)

    # Get dataset.
    if args.to_rgb is not False:
      data_kwargs = {"to_rgb": args.to_rgb}
    else:
      data_kwargs = {}
    dataset = get_data_fs(env_config, load_train=True, **data_kwargs)

    # Get data iterators.
    mean = std = None
    if args.use_dataset_stats:
      if env_config.dataset == "tiered-imagenet":
        log.info('Using tiered-imagenet stats')
        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0] 
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
    if env_config.dataset in ["matterport", "roaming-rooms"]:
      data = get_dataiter_sim(
          dataset,
          data_config,
          batch_size=config.optimizer_config.batch_size,
          nchw=mem_model.backbone.config.data_format == 'NCHW',
          seed=args.seed + restore_steps)
    else:
      data = get_dataiter_continual(
          dataset,
          data_config,
          batch_size=config.optimizer_config.batch_size,
          nchw=mem_model.backbone.config.data_format == 'NCHW',
          save_additional_info=True,
          random_box=data_config.random_box,
          seed=args.seed + restore_steps,
          da_prep2=args.da_prep2,
          mean=mean,
          std=std,
          min_object_covered=args.min_object_covered,
          )

  # Load model, training loop.
  # if args.pretrain is not None and reload_flag is None:
  backbone_is_training = True

  if args.load_memory_module is not None:
    log.info(f"Loading memory module {args.load_memory_module}")
    mem_model.load(args.load_memory_module, load_optimizer=False, load_step=False)

  if args.pretrain is not None:
    # model.load(latest_file(args.pretrain, 'weights-'))
    weights_fp = latest_file(args.pretrain, 'weights-')
    if weights_fp is None:
      weights_fp = args.pretrain
    log.info(f"Loading pretrained weights {weights_fp}")
    model.load(weights_fp, skip_fc=True, load_optimizer=False)
    if config.freeze_backbone:
      log.info('Backbone network is now frozen')
      model.set_trainable(False)  # Freeze the network.
    backbone_is_training = not config.freeze_backbone
  log.info(f"Backbone is training: {backbone_is_training}")
  if not args.eval:
    with writer.as_default() if writer is not None else dummy_context_mgr(
    ) as gs:
      train(
          mem_model,
          data['train_fs'],
          data['trainval_fs'],
          data['val_fs'],
          data['test_fs'],
          ckpt_folder,
          final_save_folder=save_folder,
          maxlen=data_config.maxlen,
          logger=logger,
          writer=writer,
          in_stage=config.in_stage,
          reload_flag=reload_flag,
          backbone_is_training=backbone_is_training)
  else:
    results_file = os.path.join(save_folder, 'results.pkl')
    logfile = os.path.join(save_folder, 'results.csv')
    if os.path.exists(results_file) and args.reeval:
      # Re-display results.
      results_all = pkl.load(open(results_file, 'rb'))
      for split, name in zip(['trainval_fs', 'val_fs', 'test_fs'],
                             ['Train', 'Val', 'Test']):
        stats = get_stats(results_all[split], tmax=data_config.maxlen)
        log_results(stats, prefix=name, filename=logfile)
    else:
      # Load the most recent checkpoint.
      if args.usebest:
        latest = latest_file(save_folder, 'best-')
      else:
        latest = latest_file(save_folder, 'weights-')

      if latest is not None:
        mem_model.load(latest)
      else:
        if args.pretrain is not None:
          latest = latest_file(args.pretrain, 'weights-')
        else:
          latest = None
        if latest is not None:
          mem_model.load(latest)
        else:
          # raise ValueError('Checkpoint not found')
          print('Checkpoint not found')
      data['trainval_fs'].reset()
      data['val_fs'].reset()
      data['test_fs'].reset()

      results_all = {}
      if args.testonly:
        split_list = ['test_fs']
        name_list = ['Test']
        nepisode_list = [config.num_episodes]
      elif args.valonly:
        split_list = ['val_fs']
        name_list = ['Val']
        nepisode_list = [config.num_episodes]
      else:
        split_list = ['trainval_fs', 'val_fs', 'test_fs']
        name_list = ['Train', 'Val', 'Test']
        # nepisode_list = [50, 50, 50]
        # nepisode_list = [10, 10, 10]
        # nepisode_list = [1, 1, 1]
        nepisode_list = [600, config.num_episodes, config.num_episodes]

      for split, name, N in zip(split_list, name_list, nepisode_list):
        data[split].reset()
        r1, nns = evaluate(mem_model, data[split], N, verbose=True)
        stats = get_stats(r1, tmax=data_config.maxlen)
        log_results(stats, prefix=name, filename=logfile, nns=nns)
        results_all[split] = (r1, nns)
      pkl.dump(results_all, open(results_file, 'wb'))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Lifelong Few-Shot Training')
  parser.add_argument('--config', type=str, default=None)
  parser.add_argument('--data', type=str, default=None)
  parser.add_argument('--env', type=str, default=None)
  parser.add_argument('--eval', action='store_true')
  parser.add_argument('--reeval', action='store_true')
  parser.add_argument('--pretrain', type=str, default=None)
  parser.add_argument('--tag', type=str, default=None)
  parser.add_argument('--testonly', action='store_true')
  parser.add_argument('--valonly', action='store_true')
  parser.add_argument('--usebest', action='store_true')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--da_prep2', action="store_true", default=False)
  parser.add_argument('--to_rgb', action="store_true", default=False)
  parser.add_argument('--use_dataset_stats', default=False, action="store_true")
  parser.add_argument('--min_object_covered', default=0.2, type=float)
  parser.add_argument('--load_memory_module', default=None)

  parser.add_argument('--file_logger', default=False, action="store_true")
  parser.add_argument('--resume', default=False, action="store_true")
  args = parser.parse_args()
  # tf.random.set_seed(1234)
  tf.random.set_seed(args.seed)
  main()

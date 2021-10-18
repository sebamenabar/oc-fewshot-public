"""Iterator for regular mini-batches.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import threading
import tensorflow as tf
import numpy as np
from numpy.random import RandomState


class MinibatchIterator(object):
  """Generates mini-batches for pretraining."""

  def __init__(self,
               dataset,
               sampler,
               batch_size,
               prefetch=True,
               preprocessor=None,
               contrastive=False,
               roaming_rooms=False,
               jitter=False,
               lrflip=False,
               seed=1):
    self._dataset = dataset
    self._preprocessor = preprocessor
    self._sampler = sampler
    sampler.set_num(dataset.get_size())
    self._contrastive = contrastive
    self.roaming_rooms = roaming_rooms
    self._jitter = jitter
    self._lrflip = lrflip
    self._mutex = threading.Lock()
    assert batch_size > 0, 'Need a positive number for batch size'
    self._batch_size = batch_size
    self._prefetch = prefetch
    self._tf_dataset = self.get_dataset()
    self._tf_dataset_iter = iter(self._tf_dataset)
    self._rnd = RandomState(seed)

  def __iter__(self):
    return self

  def reset(self):
    self.sampler.reset()
    self._tf_dataset = self.get_dataset()
    self._tf_dataset_iter = iter(self._tf_dataset)

  def get_generator(self):
    while True:
      # TODO: change the mutex lock here to TF dataset API.
      self._mutex.acquire()
      try:
        # if self.roaming_rooms:
        #   idx = self.sampler.sample_collection(1)
        # else:
        #   idx = self.sampler.sample_collection(self.batch_size)
        idx = self.sampler.sample_collection(self.batch_size)
      finally:
        self._mutex.release()
      if idx is None:
        return
      assert idx is not None
      if self.roaming_rooms:
        # x, mask, y, _ = self.dataset.get(idx[0])
        xs, masks, ys, _ = [list(x) for x in zip(*[self.dataset.get(i) for i in idx])]
        if self.jitter or self.lrflip:
          for i, (_x, _m) in enumerate(zip(xs, masks)):
            if self.jitter:
              PY = 6
              PX = 8
              x_pad_ = np.pad(
                  _x, [[PY, PY], [PX, PX], [0, 0]],
                  mode='constant',
                  constant_values=0)
              x_att_pad_ = np.pad(
                  _m, [[PY, PY], [PX, PX]],
                  mode='constant',
                  constant_values=0)
              H = _x.shape[0]
              W = _x.shape[1]

              # Jitter image and segmentation differently.
              start_y = self._rnd.randint(0, PY * 2)
              start_x = self._rnd.randint(0, PX * 2)
              start_y2 = self._rnd.randint(max(0, start_y - 2), start_y + 2)
              start_x2 = self._rnd.randint(max(0, start_x - 2), start_x + 2)
              xs[i] = x_pad_[start_y:start_y + H, start_x:start_x + W]
              masks[i] = x_att_pad_[start_y2:start_y2 +
                                        H, start_x2:start_x2 + W]
        out = {"x": tf.stack(xs), "mask": tf.expand_dims(tf.stack(masks), -1), "y": tf.stack(ys)}
        # yield {"x": x, "mask": tf.expand_dims(mask, -1), "y": y}
        yield out
      else:
        x = self.dataset.get_images(idx)
        y = self.dataset.get_labels(idx)
        yield {'x': x, 'y': y}

  def roaming_rooms_preprocess(self, data):
    _x, _m = data["x"], data["mask"]
    if self.jitter:
      PY = 6
      PX = 8
      x_pad_ = np.pad(
          _x, [[PY, PY], [PX, PX], [0, 0]],
          mode='constant',
          constant_values=0)
      x_att_pad_ = np.pad(
          _m, [[PY, PY], [PX, PX]],
          mode='constant',
          constant_values=0)
      H = _x.shape[0]
      W = _x.shape[1]

      # Jitter image and segmentation differently.
      start_y = self._rnd.randint(0, PY * 2)
      start_x = self._rnd.randint(0, PX * 2)
      start_y2 = self._rnd.randint(max(0, start_y - 2), start_y + 2)
      start_x2 = self._rnd.randint(max(0, start_x - 2), start_x + 2)
      _x = x_pad_[start_y:start_y + H, start_x:start_x + W]
      _m = x_att_pad_[start_y2:start_y2 +
                                H, start_x2:start_x2 + W]

    return {"x": _x, "mask": _m, "y": data["y"]}


  def get_dataset(self):
    """Gets TensorFlow Dataset object."""
    if self.roaming_rooms:
      shape_dict = {
          'x': tf.TensorShape([self.batch_size, 120, 160, 3]),
          'mask': tf.TensorShape([self.batch_size, 120, 160, 1]),
          'y': tf.TensorShape([None]),
          # 'x': tf.TensorShape([120, 160, 3]),
          # 'mask': tf.TensorShape([120, 160, 1]),
          # 'y': tf.TensorShape([]),
      }
      dtype_dict = {
          'x': tf.uint8,
          'mask': tf.uint8,
          'y': tf.int32,
      }
    else:
      shape_dict = {
          # 'x': tf.TensorShape([None, None, None, None]),
          'x': tf.TensorShape([self.batch_size, *self.dataset.get_images(0).shape]),
          'y': tf.TensorShape([None]),
      }
      dtype_dict = {
          'x': tf.uint8,
          'y': tf.int32,
      }
    N = self.dataset.get_size()
    ds = tf.data.Dataset.from_generator(self.get_generator, dtype_dict,
                                        shape_dict)

    def preprocess(data):
      if self.contrastive:
        data['x'] = tf.repeat(data['x'], 2, axis=0)
        data['y'] = tf.repeat(data['y'], 2, axis=0)
        # data['x2'] = self.preprocessor(data['x'])
      # else:
      if self.roaming_rooms:
        # data = self.roaming_rooms_preprocess(data)
        data["x"] = tf.concat(
          [
            self.preprocessor(data["x"]),
            # data["x"],
            # data["mask"]
          tf.cast(data["mask"], tf.float32)
          ],
          axis=-1
        )
        pass
      else:
        data['x'] = self.preprocessor(data['x'])
      return data

    ds = ds.map(preprocess)
    if self._prefetch:
      ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # if self.roaming_rooms:
    #   ds = ds.repeat().batch_size(self.batch_size, drop_remainder=True)
    return ds

  @property
  def contrastive(self):
    """Is contrastive iterator."""
    return self._contrastive

  @property
  def jitter(self):
    """Is contrastive iterator."""
    return self._jitter

  @property
  def lrflip(self):
    """Is contrastive iterator."""
    return self._lrflip

  @property
  def dataset(self):
    """Dataset object."""
    return self._dataset

  @property
  def preprocessor(self):
    """Data preprocessor."""
    return self._preprocessor

  @property
  def sampler(self):
    """Mini-batch sampler."""
    return self._sampler

  @property
  def step(self):
    """Number of steps."""
    return self.sampler._step

  @property
  def epoch(self):
    """Number of epochs."""
    return self.sampler._epoch

  @property
  def batch_size(self):
    """Batch size."""
    return self._batch_size

  @property
  def tf_dataset(self):
    return self._tf_dataset

  @property
  def cycle(self):
    return self._cycle

  @property
  def shuffle(self):
    return self._shuffle

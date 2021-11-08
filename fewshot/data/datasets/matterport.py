"""Matterport3D dataset API.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cv2
import glob
import h5py
import json
import pickle
import numpy as np
import os

from fewshot.data.registry import RegisterDataset
from fewshot.data.iterators.sim_episode_iterator import transform_bbox
from tqdm.auto import tqdm



@RegisterDataset("roaming-rooms")
@RegisterDataset("matterport")  # Legacy name
class MatterportDataset(object):

  def __init__(
      self,
      folder,
      split,
      dirpath="fewshot/data/matterport_split",
      formatted_dir=None,
    ):
    assert folder is not None
    assert split is not None
    self._folder = folder
    self._split = split
    assert split in ["train", "val", "test"]
    split_file = os.path.join(dirpath, split + '.txt')
    with open(split_file, "r") as f:
      envs = f.readlines()
    envs = set(map(lambda x: x.strip("\n"), envs))
    all_h5_files = glob.glob(os.path.join(folder, "*", "*__imgs.h5"))
    files_in_split = sorted(
        filter(lambda x: x.split("_")[-8].split(".")[0] in envs, all_h5_files))

    print(split, folder, len(files_in_split))
    assert len(files_in_split) > 0

    def make_json(x):
      return x.replace('__imgs.h5', '__annotations.json')

    basename_list = [
        ele.split("__imgs.")[0]
        for ele in files_in_split
        if os.path.exists(make_json(ele))
    ]
    self._file_list = basename_list
    # print(split, len(self._file_list))

    if formatted_dir is not None:

      self.formatted_dir = formatted_dir
      self.all_jsons = {}
      self.all_datapoints = []
      self.ys = {}

      print("Preparing dataset")
      # for basename in tqdm(self.file_list, ncols=0):

      if os.path.exists(os.path.join(formatted_dir, split + "_info.pkl")):
        with open(os.path.join(formatted_dir, split + "_info.pkl"), "rb") as f:
          data = pickle.load(f)
          self.all_jsons = data["all_jsons"]
          self.all_datapoints = data["all_datapoints"]
          self.ys = data["ys"]
      else:
        for basename in tqdm(self.file_list):
            json_fname = basename + "__annotations.json"
            with open(json_fname, "r") as f2:
                jsond = json.load(f2)

            self.all_jsons[basename] = jsond
            for i, ann in enumerate(jsond):
                for obj, info in ann.items():
                    self.all_datapoints.append((basename, i, obj))
                    # if (info["category"], info["instance_id"]) not in self.ys:
                    if (info["region_id"], info["instance_id"]) not in self.ys:
                        # self.ys[(info["category"], info["instance_id"])] = len(self.ys)
                        self.ys[(info["region_id"], info["instance_id"])] = len(self.ys)

        with open(os.path.join(formatted_dir, split + "_info.pkl"), "wb") as f:
          pickle.dump({
              "all_jsons": self.all_jsons,
              "all_datapoints": self.all_datapoints,
              "ys": self.ys
          }, f)

      self.__len = len(self.all_datapoints)
      h5 = h5py.File(os.path.join(formatted_dir, f"{self.split}.h5"), "r")
      self.rgb_dset = h5["rgb"]
      self.seg_dset = h5["seg"]
      self.rgb_len_dset = h5["rgb_len"]
      self.seg_len_dset = h5["seg_len"]
      with open(
          os.path.join(formatted_dir, f"{self.split}_mappings_fix.json"), "r"
      ) as f:
          self.mappings = json.load(f)

      self.__len = len(self.all_datapoints)

    else:
      self.__len = len(self.file_list)


  def _make_iter(self, img_arr, l_arr):
    """Makes an PNG encoding string iterator."""
    prev = 0
    l_cum = np.cumsum(l_arr)
    for i, idx in enumerate(l_cum):
      yield cv2.imdecode(img_arr[prev:idx], -1)
      prev = idx

  def get_episode(self, idx):
    """Get a single episode file."""
    basename = self.file_list[idx]
    h5_fname = basename + "__imgs.h5"
    json_fname = basename + "__annotations.json"
    with h5py.File(h5_fname, "r") as f, open(json_fname, "r") as f2:
      jsond = json.load(f2)
      inst_seg = f["instance_segmentation"][:]
      inst_seg_len = f["instance_segmentation_len"][:]
      rgb = f["matterport_RGB"][:]
      rgb_len = f["matterport_RGB_len"][:]

    rgb_iter = self._make_iter(rgb, rgb_len)
    inst_seg_iter = self._make_iter(inst_seg, inst_seg_len)
    data = []
    iter_ = enumerate(zip(rgb_iter, inst_seg_iter, jsond))
    for i, (rgb_, inst_seg_, annotation_) in iter_:
      data.append({
          "instance_seg": inst_seg_,
          "rgb": rgb_,
          "annotation": annotation_
      })
    return data

  def get_size(self):
    """Gets the total number of images."""
    # return len(self.file_list)
    return self.__len

  @property
  def folder(self):
    """Data folder."""
    return self._folder

  @property
  def split(self):
    """Data split."""
    return self._split

  @property
  def file_list(self):
    return self._file_list

  def get_label(self, idx):
    basename, i, obj = self.all_datapoints[idx]
    jsond = self.all_jsons[basename][i][obj]
    # y = self.ys[(jsond["category"], jsond["instance_id"])]
    y = self.ys[(jsond["region_id"], jsond["instance_id"])]
    return y


  def get_labels(self, inds):
    if isinstance(inds, int):
      return self.get_label(inds)
    else:
      return [self.get_label(i) for i in inds]

    
  def get_annotation(self, idx):
    basename, i, obj = self.all_datapoints[idx]
    # h5_fname = basename + "__imgs.h5"
    jsond = self.all_jsons[basename][i][obj]
    # y = self.ys[(jsond["category"], jsond["instance_id"])]

    m_basename = os.path.join(
        os.path.basename(os.path.dirname(basename)), os.path.basename(basename)
    )
    ii = self.mappings[m_basename][str(i)]

    return basename, i, obj, jsond, ii

  def get_rgb(self, ii, jsond):
    inst_seg_len = self.seg_len_dset[ii]
    rgb_len = self.rgb_len_dset[ii]

    rgb = self.rgb_dset[ii][:rgb_len]
    inst_seg = self.seg_dset[ii][:inst_seg_len]

    rgb = cv2.imdecode(rgb, -1)
    assert np.sum(rgb) != 0
    inst_seg = cv2.imdecode(inst_seg, -1)

    bbox_i = np.array(jsond["zoom_bboxes"])
    inst_id = jsond["instance_id"]

    attention_map = (inst_seg == inst_id).astype(np.uint8)

    return rgb, attention_map, bbox_i

  def get_image(self, idx, with_label=False):
    basename, i, obj, jsond, ii = self.get_annotation(idx)

    rgb, attention_map, bbox_i = self.get_rgb(ii, jsond)

    bbox_i = transform_bbox(bbox_i, attention_map.shape[:2])


    attention_bbox = bbox_i[-1]
    if np.all(attention_map == 0):
        # print(attention_bbox)

        if np.all(attention_bbox == 0):
            print("Bbox empty!")
            assert False

        attention_map = np.zeros_like(attention_map)
        y1, y2, x1, x2 = attention_bbox
        attention_map[y1:y2, x1:x2] = 1.0

    # return rgb, attention_map, attention_bbox[[0, 2, 1, 3]], y
    if with_label:
      y = self.ys[(jsond["region_id"], jsond["instance_id"])]
      return rgb, attention_map, attention_bbox, y
    return rgb, attention_map, attention_bbox # [[0, 2, 1, 3]]

  def get_images(self, inds, with_label=False):
    """Gets the image at a given index."""
    if type(inds) == int:
      return self.get_image(inds, with_label=with_label)
    else:
      images = [
        self.get_images(ind, with_label=with_label) for ind in inds
      ]
    return images


if __name__ == "__main__":
  from tqdm import tqdm
  sss_total = 0
  for sp in ['train', 'val', 'test']:
    sss = 0
    dataset = MatterportDataset("./data/matterport3d/fewshot/h5_data", sp)
    for i in tqdm(range(dataset.get_size())):
      sss += dataset.get_episode(i)
    sss_total += sss

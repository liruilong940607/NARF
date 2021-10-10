# Software License Agreement (BSD License)
#
# Copyright (c) 2019, Zerong Zheng (zzr18@mails.tsinghua.edu.cn)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the <organization> nor the
#  names of its contributors may be used to endorse or promote products
#  derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Runtime Configuration"""

from __future__ import print_function, absolute_import, division

import os

import numpy as np

# directories
dataset_dir = '/home/acc12675ut/data/dataset/THUman/dataset/'
# dataset_dir = './dataset'
# output_dir = '../../../../data/THUman_test'
bg_dir = './bg_img'  # background image directory

# axes along which flipping is applied
axis_transformation = np.array([1, -1, -1])  # which axis(axes) to flip

# output image resolution
render_img_w = 128
render_img_h = 128

# size of cornor cone added to the bouding box corners
corner_size = 0.01

# volume resolution
# volume_w = 128
# volume_h = 192
# hb_ratio = volume_w / volume_h
#
# voxel_size = 1.0 / volume_h
# corner_res = int(math.ceil(corner_size / voxel_size)) + 1

# executable program for voxelization
# binvox_dir = '../voxelizer/build/bin'

# number of images to render per mesh

# # one person
# output_dir = '/home/acc12675ut/data/dataset/THUman/single_person_train/render_128'  # single person
# data_list_fname = '/home/acc12675ut/data/dataset/THUman/single_person/data_list.txt'
# num_render_per_obj = 100  # single person
# random_lighting = False
# novel_view = False

# # one person same view test
# output_dir = '/home/acc12675ut/data/dataset/THUman/single_person_test/render_128'  # single person
# data_list_fname = '/home/acc12675ut/data/dataset/THUman/single_person/data_list.txt'
# num_render_per_obj = 20  # single person
# random_lighting = False
# novel_view = False

# # one person novel view
# output_dir = '/home/acc12675ut/data/dataset/THUman/single_person_novel_view_test/render_128'  # single person
# data_list_fname = '/home/acc12675ut/data/dataset/THUman/single_person/data_list.txt'
# num_render_per_obj = 20  # single person
# random_lighting = False
# novel_view = True

# # one person female
# output_dir = '/home/acc12675ut/data/dataset/THUman/single_person_female/train/render_128'  # single person
# data_list_fname = '/home/acc12675ut/data/dataset/THUman/single_person_female/data_list.txt'
# num_render_per_obj = 100  # single person
# random_lighting = False
# novel_view = False

# # one person female same test
# output_dir = '/home/acc12675ut/data/dataset/THUman/single_person_female/test/render_128'  # single person
# data_list_fname = '/home/acc12675ut/data/dataset/THUman/single_person_female/data_list.txt'
# num_render_per_obj = 20  # single person
# random_lighting = False
# novel_view = False

# # one person female novel view
# output_dir = '/home/acc12675ut/data/dataset/THUman/single_person_female/novel_view_test/render_128'  # single person
# data_list_fname = '/home/acc12675ut/data/dataset/THUman/single_person_female/data_list.txt'
# num_render_per_obj = 20  # single person
# random_lighting = False
# novel_view = True

# # all person
# output_dir = '/home/acc12675ut/data/dataset/THUman/all_person/render_128'  # all person
# data_list_fname = '/home/acc12675ut/data/dataset/THUman/dataset/data_list.txt'
# num_render_per_obj = 10  # all person
# random_lighting = True
# novel_view = False

# # all person test
# output_dir = '/home/acc12675ut/data/dataset/THUman/all_person_test/render_128'  # all person
# data_list_fname = '/home/acc12675ut/data/dataset/THUman/dataset/data_list.txt'
# num_render_per_obj = 1  # all person
# random_lighting = True
# novel_view = False

# # all person novel view
# output_dir = '/home/acc12675ut/data/dataset/THUman/all_person_novel_view_test/render_128'  # all person
# data_list_fname = '/home/acc12675ut/data/dataset/THUman/dataset/data_list.txt'
# num_render_per_obj = 1  # all person
# random_lighting = True
# novel_view = True
# autoencoder = True

# # one person release test
# output_dir = '/home/acc12675ut/data/dataset/THUman/release_test/render_128'  # single person
# data_list_fname = '/home/acc12675ut/data/dataset/THUman/single_person/data_list.txt'
# num_render_per_obj = 100  # single person
# random_lighting = False
# novel_view = False

# # one person same view test
# output_dir = '/home/acc12675ut/data/dataset/THUman/release_test_test/render_128'  # single person
# data_list_fname = '/home/acc12675ut/data/dataset/THUman/single_person/data_list.txt'
# num_render_per_obj = 20  # single person
# random_lighting = False
# novel_view = False

# one person novel view
# output_dir = '/home/acc12675ut/data/dataset/THUman/release_test_novel_view_test/render_128'  # single person
data_list_fname = '/home/acc12675ut/data/dataset/THUman/single_person/data_list.txt'
# num_render_per_obj = 20  # single person
random_lighting = False
# novel_view = True

output_root_dir = '/home/acc12675ut/data/dataset/THUman/release'

render_setting = {
    "train": {"output_dir": os.path.join(output_root_dir, "train", "render_" + str(render_img_w)),
              "num_render_per_obj": 100, "novel_view": False},
    "same_view": {"output_dir": os.path.join(output_root_dir, "same_view", "render_" + str(render_img_w)),
                  "num_render_per_obj": 20, "novel_view": False},
    "novel_view": {"output_dir": os.path.join(output_root_dir, "novel_view", "render_" + str(render_img_w)),
                   "num_render_per_obj": 20, "novel_view": True},
}

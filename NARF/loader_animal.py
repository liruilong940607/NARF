import logging
import os
import json
from typing import List

import torch
import torch.nn.functional as F
import numpy as np
import imageio


LOGGER = logging.getLogger(__name__)


class AnimalSubjectParser():
    """Single subject data parser."""

    SPLIT = ["all", "train", "test", "test_ood"]

    def __init__(
        self,
        subject_id: str = "Hare_male_full_RM",
        root_fp: str = "/home/ruilongli/workspace/blenderlib/results_multi_action5/",
        split_test_actions: List[str] = ["Death_1", "Death_2", "Death_Sink"],
        split: str = "all",
    ):
        assert split in AnimalSubjectParser.SPLIT 
        self.id = subject_id
        self.split = split
        self.split_test_actions = split_test_actions
        self.root_fp = root_fp
        self.dtype = torch.get_default_dtype()
        self.root_dir = os.path.join(root_fp, subject_id)

        actions = sorted([
            fp for fp in os.listdir(self.root_dir)
            if os.path.exists(os.path.join(self.root_dir, fp, "camera.json"))
        ])
        self.actions = {
            "all": actions,
            "train": list(set(actions) - set(split_test_actions)),
            "test": list(set(actions) - set(split_test_actions))[0:1],
            "test_ood": split_test_actions,
        }[split]

        self.data_list = self.get_data_list()
        LOGGER.info("Dataset Created! N samples: %d", self.__len__())

    def __len__(self):
        return len(self.data_list)

    def get_data_list(self):
        data_list = []
        for action in self.actions:
            data = self.load_camera(action)
            frame_ids = list(data.keys())
            for frame_id in frame_ids:
                camera_ids = list(data[frame_id].keys())
                data_list += [(action, frame_id, camera_id) for camera_id in camera_ids]
        return data_list

    def load_camera(self, action, frame_id = None, camera_id = None):
        path = os.path.join(self.root_dir, action, "camera.json")
        with open(path, mode="r") as fp:
            data = json.load(fp)
        if frame_id is not None and camera_id is not None:
            K = torch.tensor(data[frame_id][camera_id]["intrin"]).to(self.dtype)
            w2c = torch.tensor(data[frame_id][camera_id]["extrin"]).to(self.dtype)
            return K, w2c  # shape [3, 3], [4, 4]
        else:
            return data

    def load_image(self, action, frame_id, camera_id):
        path = os.path.join(
            self.root_dir, action, "image", camera_id, "%08d.png" % int(frame_id)
        )
        image = torch.from_numpy(imageio.imread(path)).to(self.dtype)
        return image  # shape [800, 800, 4], value 0 ~ 255

    def load_pose(self, action, frame_id):
        path = os.path.join(self.root_dir, action, "meta_data.npz")
        frame_id = int(frame_id)
        with open(path, mode="rb") as fp:
            data = np.load(fp)
            # idxs for real bones (exclude helper bones such as root)
            real_bones_idxs = np.where(np.max(data["weights"], axis=0) > 0)[0].tolist()
            # bone matrics: bone space (head at origin) -> view space [J, 4, 4]
            pose_matrix = torch.from_numpy(data["pose_matrix"][frame_id]).to(self.dtype)
            # bone matrics: bone space (head at origin) -> canonical space [J, 4, 4]
            rest_matrix = torch.from_numpy(data["rest_matrix"]).to(self.dtype)
            # bone locs in canonical space [J, 3], [J, 3]
            rest_loc_head = torch.from_numpy(data["rest_loc_head"]).to(self.dtype)
            rest_loc_tail = torch.from_numpy(data["rest_loc_tail"]).to(self.dtype)
            if False:  # check sanity
                assert torch.allclose(rest_matrix[:, :3, 3], rest_loc_head)
                bone_loc_head = torch.einsum(
                    "bij,bj->bi", 
                    rest_matrix.inverse()[:, :3, :4], 
                    F.pad(rest_loc_head, (0, 1), value=1)
                )
                assert torch.allclose(bone_loc_head, torch.zeros_like(bone_loc_head), atol=1e-6)
                bone_loc_tail = torch.einsum(
                    "bij,bj->bi", 
                    rest_matrix.inverse()[:, :3, :4], 
                    F.pad(rest_loc_tail, (0, 1), value=1)
                )
                # in bone space, the head is at origin, the tail is at [0, +y, 0]
                assert torch.allclose(
                    bone_loc_tail[:, [0, 2]], torch.zeros_like(bone_loc_tail[:, [0, 2]]), atol=1e-6
                )
            
        return {
            "pose_matrix": pose_matrix[real_bones_idxs],
            "rest_matrix": rest_matrix[real_bones_idxs],
            "rest_loc_head": rest_loc_head[real_bones_idxs],
            "rest_loc_tail": rest_loc_tail[real_bones_idxs],
        }
        
    def estimate_bone_space_bbox(self):
        # In each bone space, calculate an effective bounding-box by 
        # accounting those verts that listen to this bone.
        path = os.path.join(parser.root_dir, parser.actions[0], "meta_data.npz")
        with open(path, mode="rb") as fp:
            data = np.load(fp)
            real_bones_idxs = np.where(np.max(data["weights"], axis=0) > 0)[0].tolist()
            rest_matrix = torch.from_numpy(data["rest_matrix"]).float()
            rest_verts = torch.from_numpy(data["rest_verts"]).float()
            lbs_weights = torch.from_numpy(data["weights"]).float()
        
        bboxs = []
        for _, bone_id in enumerate(real_bones_idxs):
            bone_verts = torch.einsum(
                "ij,bj->bi",
                rest_matrix[bone_id, :3, :4],
                F.pad(rest_verts[lbs_weights[:, bone_id] > 0], (0, 1), value=1)
            )
            bboxs.append(
                torch.cat([bone_verts.min(dim=0).values, bone_verts.max(dim=0).values])
            )
        bboxs = torch.stack(bboxs).reshape(-1, 3, 2)
        bboxs_min = bboxs[:, :, 0]  # [J', 3]
        bboxs_max = bboxs[:, :, 0]  # [J', 3]
        return bboxs_min, bboxs_max


if __name__ == "__main__":
    parser = AnimalSubjectParser()

    # In each bone space, calculate an effective bounding-box by 
    # accounting those verts that listen to this bone.
    path = os.path.join(parser.root_dir, parser.actions[0], "meta_data.npz")
    with open(path, mode="rb") as fp:
        data = np.load(fp)
        real_bones_idxs = np.where(np.max(data["weights"], axis=0) > 0)[0].tolist()
        rest_matrix = torch.from_numpy(data["rest_matrix"]).float()
        rest_verts = torch.from_numpy(data["rest_verts"]).float()
        lbs_weights = torch.from_numpy(data["weights"]).float()
    
    bboxs = []
    for i, bone_id in enumerate(real_bones_idxs):
        bone_verts = torch.einsum(
            "ij,bj->bi",
            rest_matrix[bone_id, :3, :4],
            F.pad(rest_verts[lbs_weights[:, bone_id] > 0], (0, 1), value=1)
        )
        bboxs.append(
            torch.cat([bone_verts.min(dim=0).values, bone_verts.max(dim=0).values])
        )
    bboxs = torch.stack(bboxs).reshape(-1, 3, 2)
    bboxs = torch.cat([bboxs[:, :, 0].min(dim=0).values, bboxs[:, :, 1].max(dim=0).values])
    print (bboxs)

    for action, fid, cid in parser.data_list:
        K, w2c = parser.load_camera(action, fid, cid)
        c2w = w2c.inverse()
        cam_pos = c2w[:3, 3]
        pose_matrix = parser.load_pose(action, fid)["pose_matrix"]
        cam_pos_bone = pose_matrix.inverse()[:3, :4] @ F.pad(cam_pos, (0, 1), value=1)


    # action, frame_id, _ = parser.data_list[100]
    # parser.load_pose(action, frame_id)

    # action = parser.actions[0]
    # path = os.path.join(parser.root_dir, action, "meta_data.npz")
    # with open(path, mode="rb") as fp:
    #     data = np.load(fp)
    #     rest_verts = data["rest_verts"]
    # print (rest_verts.reshape(-1, 3).min(axis=0))
    # print (rest_verts.reshape(-1, 3).max(axis=0))
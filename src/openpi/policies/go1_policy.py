# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Policy transforms for the Go1 robot."""
import dataclasses
from typing import ClassVar
import numpy as np
import torch
import copy
import openpi.models.model as _model
import openpi.transforms as transforms
@dataclasses.dataclass(frozen=True)
class Go1Inputs(transforms.DataTransformFn):
    """Inputs for the Go1 policy.
    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [32]
    - actions: [action_horizon, 22]
    """
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0
    state_mask: np.ndarray | None = None
    action_mask: np.ndarray | None = None
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_head", "hand_left", "hand_right")
    rename_map = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb"
    }
    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0
        state = transforms.pad_to_dim(data["state"], self.action_dim)
        state = copy.deepcopy(state)
        if len(state) in [186, 190]:
            indices = list(range(54, 68)) + [0, 1] + list(range(2, 54)) + list(range(68, len(state)))
            state = state[indices]
        if len(state)>len(self.state_mask):
            state = state[:len(self.state_mask)]
        if self.state_mask is not None:
            state[self.state_mask] = 0
        state = state.squeeze()
        images = {}
        for camera in self.EXPECTED_CAMERAS:
            if camera in data["images"]:
                img = data["images"][camera]
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                if np.issubdtype(img.dtype, np.floating):
                    img = (255 * img).astype(np.uint8)
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                images[self.rename_map[camera]] = img
            else:
                raise ValueError(f"Camera {camera} not found in data")
        image_mask = {self.rename_map[camera]: np.True_ for camera in self.EXPECTED_CAMERAS}
        inputs = {
            "image": images,
            "image_mask": image_mask,
            "state": state,
        }
        if "actions" in data:
            actions = data["actions"]
            if actions.shape[1] in [36, 40]:
                actions = np.column_stack((actions[:, 16:30], actions[:, 0], actions[:, 1]))
                if actions.shape[1] > len(self.action_mask):
                    actions = actions[:,:len(self.action_mask)]
            if self.action_mask is not None:
                actions[:, self.action_mask[:actions.shape[1]]] = 0
            actions = transforms.pad_to_dim(actions, self.action_dim)
            inputs["actions"] = actions.squeeze()
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs
@dataclasses.dataclass(frozen=True)
class Go1Outputs(transforms.DataTransformFn):
    """Outputs for the Go1 policy."""
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :22])}

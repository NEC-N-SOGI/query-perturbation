from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torchvision.transforms import Resize

from query_perturbation.data.box_loader import BoxInfo, ImageBoxLoader
from query_perturbation.models.weight import BaseWeight


class QPert(ABC):
    @abstractmethod
    def __init__(
        self,
        config_path: Path,
        n_pc: float | int,
        device: str | torch.device,
        weighter: Optional[BaseWeight],
    ) -> None:
        # dummy
        self.device = device
        self.n_pc = n_pc
        self.resize = Resize((0, 0))
        self.kernel = torch.zeros((0, 0), device="cuda")
        self.stride = 0
        self.weighter = weighter
        self.model = torch.nn.Module()

        pass

    @abstractmethod
    def get_key_subspace(
        self, imgs: Tensor, boxs: list[BoxInfo]
    ) -> tuple[Tensor, list[Tensor], list[Tensor], list[Tensor]]:
        pass

    @abstractmethod
    def extract_vis_feats(
        self,
        image: Tensor,
        box_infos: Optional[list[BoxInfo]] = None,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def extract_text_feats(
        self, texts: list[str], ret_options: bool = False
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        pass

    @abstractmethod
    def forward(self, loader: ImageBoxLoader, verbosity: int = 0) -> Tensor:
        pass

    @abstractmethod
    def apply_itm(
        self,
        image: Tensor,
        text_ids: Tensor,
        text_atts: Tensor,
        box_infos: Optional[BoxInfo] = None,
    ) -> Tensor:
        pass

    def input_layer_size(
        self, model: torch.nn.Module, input_img_size: tuple
    ) -> None:
        """get info of the input patch embed layer."""
        self.k_size = model.visual_encoder.patch_embed.proj.kernel_size
        self.stride = model.visual_encoder.patch_embed.proj.stride[0]
        self.kernel = (
            torch.ones(self.k_size, device="cuda").unsqueeze(0).unsqueeze(0)
        )
        self.resize = Resize(input_img_size)

    def get_key_weight(
        self,
        boxs: BoxInfo,
    ) -> Tensor | None:
        """gen key_weight. currently 1 to a key including a box. otherwise set to 0"""
        weights = []
        img_size = boxs.img_size
        for box in boxs.boxes:
            mask = torch.zeros(img_size, device="cuda")
            mask[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = 1

            x = self.resize(mask.unsqueeze(0).unsqueeze(0))
            x[x > 0] = 1

            # convolve x with kernel
            k_size = self.kernel.shape[2:]
            x = torch.nn.functional.conv2d(x, self.kernel, stride=self.stride)
            x = x / (k_size[0] * k_size[1])

            cls_token_w = torch.zeros((1), device="cuda")

            weights.append(torch.cat([cls_token_w, x.flatten(2).squeeze()], 0))

        if len(weights) > 0:
            return torch.stack(weights, 0)

        # No box
        return None

    def roi_pooling(
        self,
        key_feats: Tensor,
        boxs: list[BoxInfo],
    ) -> list[Tensor]:
        """
        Args:
            key_feats (Tensor): [n_img, ...]
            boxs (list[BoxInfo]): len(boxs) = n_imgs

        Returns:
            Tensors: pooled keys. len(retunr) = n_imgs
        """
        assert key_feats.shape[0] == len(
            boxs
        ), "Length of key_feats, and boxs must be the same"

        pooled_keys = []
        for keys, box in zip(key_feats, boxs):
            key_weight = self.get_key_weight(box)

            if key_weight is None:
                # no bbox
                key_weight = torch.empty(0, device="cuda")
            else:
                key_weight = (
                    key_weight.unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(4)
                    .expand((key_weight.shape[0], *keys.shape))
                )

            pooled_keys.append(keys * key_weight.to(key_feats.dtype))

        return pooled_keys

    def get_n_pc(self, eig_val: Tensor) -> Tensor:
        """get select subspace basis"""
        n_pc = self.n_pc
        ev_sel = (eig_val.cumsum(-1) / eig_val.sum(-1, keepdim=True)) < n_pc
        one_dim = torch.zeros_like(ev_sel)

        # if n_pc < 1, use cumulative ratio
        if n_pc < 1:
            one_dim[..., 0] = True
            return ev_sel | one_dim  # at least 1-dimension_
        elif isinstance(n_pc, int):
            one_dim[..., :n_pc] = True
        else:
            raise ValueError("n_pc must be int or float")
        return one_dim

    def get_projection_mat(
        self, eig_val: Tensor, eig_vec: Tensor, box_info: Optional[BoxInfo] = None
    ) -> Tensor:
        """calculate projection matricies"""
        if len(eig_val) == 0:
            return torch.empty(0, device="cuda")

        ev_sel = self.get_n_pc(eig_val)
        _vec = eig_vec.clone()
        _vec[~ev_sel] = 0

        # #obj, #layer, #head, #dim, #dim
        p_mat = _vec.transpose(-1, -2) @ _vec

        if box_info is not None and self.weighter is not None:
            # alpha.
            weight = self.weighter(box_info).to(p_mat)
            diff = len(p_mat.shape) - len(weight.shape)
            for _ in range(diff):
                weight = weight.unsqueeze(-1)
            p_mat = p_mat * weight

        return p_mat

    def pca(
        self, feats: Tensor, box_info: Optional[BoxInfo] = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        _feats = feats.float()
        auto_corr = _feats.transpose(-2, -1) @ _feats
        d, v = torch.linalg.eigh(auto_corr)

        dtype = feats.dtype
        d, v = d.to(dtype), v.to(dtype)

        eig_val, eig_vec = torch.flip(d, [-1]), torch.flip(v, [-1])
        proj_mat = self.get_projection_mat(eig_val, eig_vec, box_info)

        return eig_val, eig_vec, proj_mat

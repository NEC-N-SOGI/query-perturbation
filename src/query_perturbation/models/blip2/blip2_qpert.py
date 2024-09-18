from pathlib import Path
from typing import Optional

import torch
import tqdm
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import Resize

from query_perturbation.data.box_loader import BoxInfo, ImageBoxLoader
from query_perturbation.lavis_utils import get_model
from query_perturbation.models.base_qpert import QPert
from query_perturbation.models.blip2.patching_blip2 import (
    monkey_patch_crossattention,
)
from query_perturbation.models.weight import BaseWeight


class BLIP2Extractor(QPert):
    def __init__(
        self,
        config_path: Path,
        n_pc: float | int,
        device: str | torch.device,
        weighter: Optional[BaseWeight] = None,
        do_normalize: bool = False,
        dtype: torch.dtype = torch.float32,
        apply_qpert: bool = True,
    ) -> None:
        self.n_pc = n_pc
        self.device = device
        self.do_normalize = do_normalize
        self.weighter = weighter
        self.apply_qpert = apply_qpert

        # model loading
        cfg, _, self.model = get_model(str(config_path))
        img_size = cfg.config["model"]["image_size"]
        self.img_size = img_size

        if apply_qpert:
            monkey_patch_crossattention(self.model, do_normalize)

        self.model.to(device).to(dtype)

        # params for key pooling
        lavis_data_cfg = list(cfg.datasets_cfg.values())[0]
        new_size = lavis_data_cfg["vis_processor"]["eval"]["image_size"]

        self.k_size = self.model.visual_encoder.patch_embed.proj.kernel_size
        self.stride = self.model.visual_encoder.patch_embed.proj.stride[0]
        self.kernel = (
            torch.ones(self.k_size, device="cuda").unsqueeze(0).unsqueeze(0)
        )
        self.resize = Resize((new_size, new_size))

        self.input_layer_size(self.model, (new_size, new_size))

    def extract_feat_map(self, imgs: Tensor) -> Tensor:
        vit_feat: Tensor
        _, vit_feat = self.model.forward_image(imgs)

        return vit_feat

    def extract_key_feats(self, feat_map: Tensor) -> Tensor:
        image_atts = torch.ones(feat_map.size()[:-1], dtype=torch.long).to(
            feat_map.device
        )

        query_tokens = self.model.query_tokens.expand(feat_map.shape[0], -1, -1)

        _ = self.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=feat_map,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        key_feats = []
        for layer in self.model.Qformer.bert.encoder.layer:
            if hasattr(layer, "crossattention"):
                key_feats.append(layer.crossattention.self.keys.unsqueeze(1))

        return torch.cat(key_feats, 1)

    def _get_pmat(
        self,
        image: Tensor,
        box_infos: Optional[list[BoxInfo]] = None,
    ) -> tuple[Tensor, list[Tensor] | None]:
        if box_infos is not None and self.apply_qpert:
            if image.dim() == 3:
                image = image.unsqueeze(0)
            feat_map, _, _, p_mat = self.get_key_subspace(image, box_infos)
            model_input = feat_map
        else:
            model_input = self.extract_feat_map(image)
            p_mat = None

        return model_input, p_mat

    def extract_vis_feats(
        self,
        image: Tensor,
        box_infos: Optional[list[BoxInfo]] = None,
    ) -> Tensor:
        self.clear_proj_mat()

        # get pmat and feature map
        model_input, p_mat = self._get_pmat(image, box_infos)
        if p_mat is not None:
            self.set_proj_mat(p_mat)

        # prepare
        model = self.model
        dtype = list(model.parameters())[0].dtype
        model_input = model_input.to(model.device).to(dtype)

        image_atts = torch.ones(
            model_input.size()[:-1], dtype=torch.long, device=model.device
        )
        query_tokens = model.query_tokens.expand(model_input.shape[0], -1, -1).to(
            model.device
        )

        # forward qformer w qpert
        query_output = model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=model_input,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_feat = query_output.last_hidden_state

        image_embed: Tensor = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        return image_embed

    def extract_text_feats(
        self, texts: list[str], ret_options: bool = False
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        model = self.model
        num_text = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []
        text_atts = []
        for i in range(0, num_text, text_bs):
            text = texts[i : min(num_text, i + text_bs)]
            text_input = model.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            ).to(model.device)
            # Q-former output
            text_feat = model.forward_text(text_input)

            # Inpu
            text_embed = F.normalize(model.text_proj(text_feat))
            text_embeds.append(text_embed.cpu())
            text_ids.append(text_input.input_ids)
            text_atts.append(text_input.attention_mask)

        if ret_options:
            return (
                torch.cat(text_embeds, dim=0),
                torch.cat(text_ids, dim=0),
                torch.cat(text_atts, dim=0),
            )
        return torch.cat(text_embeds, dim=0)

    def set_proj_mat(self, p_mat: list[Tensor]) -> None:
        """

        Args:
            p_mat (list[Tensor]): len = n_batch, each element is (#obj, #layer, #head #dim, #dim)
            #obj is optional
        """
        cnt = 0
        try:
            dim = max([p.dim() for p in p_mat if len(p) > 0])
        except ValueError:
            dim = 5

        for layer in self.model.Qformer.bert.encoder.layer:
            if hasattr(layer, "crossattention"):
                if dim == 5:
                    _p = [p[:, cnt] if len(p) > 0 else [] for p in p_mat]
                else:
                    _p = [p[cnt] if len(p) > 0 else [] for p in p_mat]
                layer.crossattention.self.query.set_projection_mat(_p)
                cnt += 1

    def clear_proj_mat(self) -> None:
        for layer in self.model.Qformer.bert.encoder.layer:
            if hasattr(layer, "crossattention"):
                layer.crossattention.self.query.projection_mat = []

    def apply_itm(
        self,
        image: Tensor,
        text_ids: Tensor,
        text_atts: Tensor,
        box_infos: Optional[BoxInfo] = None,
    ) -> Tensor:
        self.clear_proj_mat()

        _box_infos = [box_infos] if box_infos is not None else None
        model_input, p_mat = self._get_pmat(image, _box_infos)

        bs = text_ids.shape[0]
        model_input = model_input.repeat(bs, 1, 1)

        if p_mat is not None:
            p_mat = p_mat * bs
            self.set_proj_mat(p_mat)

        score: Tensor = self.model.compute_itm(
            image_inputs=model_input,
            text_ids=text_ids,
            text_atts=text_atts,
        )

        return score

    def get_key_subspace(
        self, imgs: Tensor, boxs: list[BoxInfo]
    ) -> tuple[Tensor, list[Tensor], list[Tensor], list[Tensor]]:
        feat_map = self.extract_feat_map(imgs)
        key_feats = self.extract_key_feats(feat_map)

        pooled_keys = self.roi_pooling(key_feats, boxs)

        eig_vals, eig_vecs, proj_mats = [], [], []
        for i, keys in enumerate(pooled_keys):
            if keys.shape[0] == 0:
                eig_vals.append(torch.empty(0, device="cuda"))
                eig_vecs.append(torch.empty(0, device="cuda"))
                proj_mats.append(torch.empty(0, device="cuda"))
                continue

            eig_val, eig_vec, proj_mat = self.pca(keys, boxs[i])
            eig_vals.append(eig_val)
            eig_vecs.append(eig_vec)
            proj_mats.append(proj_mat)

        return feat_map, eig_vals, eig_vecs, proj_mats

    def forward(self, loader: ImageBoxLoader, verbosity: int = 0) -> Tensor:
        n_imgs = len(loader.loader.loader.dataset)
        first = True

        if verbosity > 0:
            loader = tqdm.tqdm(loader)

        for images, ids, box_infos in loader:
            _feat = self.extract_vis_feats(images, box_infos=box_infos)
            if first:
                feats = torch.empty(
                    (n_imgs, *_feat.shape[1:]),
                    device=_feat.device,
                    dtype=_feat.dtype,
                )
                first = False
            feats[ids] = _feat

        return feats

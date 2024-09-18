import torch
from torch import Tensor
from tqdm import tqdm

from query_perturbation.base_task import BaseTask
from query_perturbation.data.box_loader import ImageBoxLoader
from query_perturbation.models.blip2.blip2_qpert import BLIP2Extractor


class BLIP2Task(BaseTask):
    def __init__(
        self,
        loader: ImageBoxLoader,
        extractor: BLIP2Extractor,
        use_itm: bool = False,
        top_k: int = 64,
    ):
        super().__init__(loader, extractor)
        self.use_itm = use_itm
        self.top_k = top_k

    def calc_txt_img_sim(
        self, image_embeds: Tensor, text_feats: Tensor
    ) -> tuple[Tensor, Tensor]:
        _sim = image_embeds.view((-1, image_embeds.shape[-1])) @ text_feats.T
        i2t_sim = (_sim.view((*image_embeds.shape[:2], -1)).max(1)[0]).squeeze()

        if not self.use_itm:
            return i2t_sim, i2t_sim.T

        return self.itm_reranking(i2t_sim, self.loader.dataset.text, self.top_k)

    def itm_reranking(
        self, i2t_itc_sim: Tensor, texts: list[str], k_test: int
    ) -> tuple[Tensor, Tensor]:
        _, text_ids, text_atts = self.extractor.extract_text_feats(texts, True)

        #
        t2i_ranks = torch.sort(i2t_itc_sim.T, descending=True).indices[:, :k_test]
        t2i_convert: dict[int, list[int]] = {}
        for i, idx in enumerate(t2i_ranks):
            for j in idx:
                img_idx = j.item()
                if img_idx in list(t2i_convert.keys()):
                    t2i_convert[img_idx].append(i)
                else:
                    t2i_convert[img_idx] = [i]
        n_queries, n_targets = i2t_itc_sim.shape

        i2t_sims = torch.full(
            (n_queries, n_targets), -100.0, dtype=i2t_itc_sim.dtype
        )
        t2i_sims = torch.full(
            (n_targets, n_queries), -100.0, dtype=i2t_itc_sim.dtype
        )

        bs = 20
        for images, idx, box_infos in tqdm(self.loader):
            for image, _id_pt, box_info in zip(images, idx, box_infos):
                _id = _id_pt.item()
                # i2t
                sims = i2t_itc_sim[_id]
                topk_sim, topk_idx = sims.topk(k=k_test, dim=0)

                for k_batch in range(0, len(topk_idx), bs):
                    _topk_idx = topk_idx[k_batch : min(k_batch + bs, len(topk_idx))]

                    score = self.extractor.apply_itm(
                        image,
                        text_ids[_topk_idx],
                        text_atts[_topk_idx].squeeze(),
                        box_infos=box_info,
                    )

                    i2t_sims[_id, _topk_idx] = (
                        score.cpu()
                        + topk_sim[k_batch : min(k_batch + bs, len(topk_idx))].cpu()
                    )

                # t2i
                t2i_topk_idx = t2i_convert[_id]
                topk_sim = i2t_itc_sim[_id, t2i_topk_idx]

                for k_batch in range(0, len(t2i_topk_idx), bs):
                    _topk_ids = t2i_topk_idx[
                        k_batch : min(k_batch + bs, len(t2i_topk_idx))
                    ]

                    _text_ids = text_ids[_topk_ids]
                    _text_atts = text_atts[_topk_ids]

                    if _text_ids.dim() == 1:
                        _text_ids = _text_ids.unsqueeze(0)
                    if _text_atts.dim() == 1:
                        _text_atts = _text_atts.unsqueeze(0)

                    score = self.extractor.apply_itm(
                        image,
                        _text_ids,
                        _text_atts,
                        box_infos=box_info,
                    )

                    t2i_sims[_topk_ids, _id] = (
                        score.cpu()
                        + topk_sim[
                            k_batch : min(k_batch + bs, len(t2i_topk_idx))
                        ].cpu()
                    )

        return i2t_sims.cpu(), t2i_sims.cpu()

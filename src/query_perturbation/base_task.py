from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from query_perturbation.data.box_loader import ImageBoxLoader
from query_perturbation.models.base_qpert import QPert


@dataclass(frozen=True)
class EvalMetrics:
    txt_r1: float
    txt_r5: float
    txt_r10: float
    txt_r_mean: float
    txt_mdr: float
    txt_mer: float
    txt_mrr: float
    img_r1: float
    img_r5: float
    img_r10: float
    img_r_mean: float
    img_mdr: float
    img_mer: float
    img_mrr: float
    r_mean: float
    agg_metrics: float

    t2i_ranks: np.ndarray
    i2t_ranks: np.ndarray


class BaseTask(ABC):
    def __init__(self, loader: ImageBoxLoader, extractor: QPert, verbosity: int = 1):
        self.loader = loader
        extractor.model.to(loader.dtype)
        self.extractor = extractor
        self.verbosity = verbosity

    @abstractmethod
    def calc_txt_img_sim(
        self, image_embeds: Tensor, text_feats: Tensor
    ) -> tuple[Tensor, Tensor]:
        pass

    def compute_metrics(
        self, i2t_ranks: np.ndarray, t2i_ranks: np.ndarray
    ) -> EvalMetrics:
        # Compute metrics
        tr1 = 100.0 * len(np.where(i2t_ranks < 1)[0]) / len(i2t_ranks)
        tr5 = 100.0 * len(np.where(i2t_ranks < 5)[0]) / len(i2t_ranks)
        tr10 = 100.0 * len(np.where(i2t_ranks < 10)[0]) / len(i2t_ranks)
        tmdr = np.median(i2t_ranks + 1)
        tmer = np.mean(i2t_ranks + 1)
        tmrr = np.mean(1 / (i2t_ranks + 1))

        # Compute metrics
        ir1 = 100.0 * len(np.where(t2i_ranks < 1)[0]) / len(t2i_ranks)
        ir5 = 100.0 * len(np.where(t2i_ranks < 5)[0]) / len(t2i_ranks)
        ir10 = 100.0 * len(np.where(t2i_ranks < 10)[0]) / len(t2i_ranks)
        imdr = np.median(t2i_ranks + 1)
        imer = np.mean(t2i_ranks + 1)
        imrr = np.mean(1 / (t2i_ranks + 1))

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        agg_metrics = (tr1 + tr5 + tr10) / 3

        return EvalMetrics(
            txt_r1=tr1,
            txt_r5=tr5,
            txt_r10=tr10,
            txt_r_mean=tr_mean,
            txt_mdr=tmdr,
            txt_mer=tmer,
            txt_mrr=tmrr,
            img_r1=ir1,
            img_r5=ir5,
            img_r10=ir10,
            img_r_mean=ir_mean,
            img_mdr=imdr,
            img_mer=imer,
            img_mrr=imrr,
            r_mean=r_mean,
            agg_metrics=agg_metrics,
            t2i_ranks=t2i_ranks,
            i2t_ranks=i2t_ranks,
        )

    def report_metircs_on_size(
        self,
        i2t_ranks: np.ndarray,
        t2i_ranks: np.ndarray,
        areas: np.ndarray,
        thresholds: np.ndarray,
        img2txt: dict[int, list[int]],
    ) -> list[EvalMetrics]:
        expanded_area = np.zeros_like(t2i_ranks)
        for k, v in img2txt.items():
            target_area = areas[k]
            for i in v:
                expanded_area[i] = target_area

        i2t_indexes = []
        t2i_indexes = []
        for i in range(len(thresholds) - 1):
            small_th, large_th = thresholds[i], thresholds[i + 1]
            i2t_indexes.append(np.where((small_th <= areas) & (areas < large_th))[0])
            t2i_indexes.append(
                np.where((small_th <= expanded_area) & (expanded_area < large_th))[0]
            )

        return [
            self.compute_metrics(i2t_ranks[i2t], t2i_ranks[t2i])
            for i2t, t2i in zip(i2t_indexes, t2i_indexes)
        ]

    def report_metrics(
        self,
        scores_i2t: np.ndarray,
        scores_t2i: np.ndarray,
        txt2img: dict[int, int | list[int]],
        img2txt: dict[int, list[int]],
        areas: np.ndarray,
        thresholds: np.ndarray = np.linspace(0, 1.0, 11),
    ) -> tuple[EvalMetrics, list[EvalMetrics]]:
        # Images->Text
        i2t_ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            if isinstance(img2txt[index], int):
                rank = np.where(inds == img2txt[index])[0][0]
            else:
                for i in img2txt[index]:
                    tmp = np.where(inds == i)[0][0]
                    if tmp < rank:
                        rank = tmp
            i2t_ranks[index] = rank

        # Text->Images
        t2i_ranks = np.zeros(scores_t2i.shape[0])

        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            t2i_ranks[index] = np.where(inds == txt2img[index])[0][0]

        org_metrics = self.compute_metrics(i2t_ranks, t2i_ranks)

        metrics_on_size = self.report_metircs_on_size(
            i2t_ranks, t2i_ranks, areas, thresholds, img2txt
        )
        return org_metrics, metrics_on_size

    def run_retrieval(self) -> tuple[Tensor, Tensor]:
        with torch.inference_mode():
            image_embeds = self.extractor.forward(
                self.loader, verbosity=self.verbosity
            )
            text_feats = self.extractor.extract_text_feats(self.loader.dataset.text)
            if isinstance(text_feats, tuple):
                text_feats = text_feats[0]
            text_feats = text_feats.to(image_embeds.device)

            i2t_sim, t2i_sim = self.calc_txt_img_sim(image_embeds, text_feats)

        return i2t_sim, t2i_sim

    def run(
        self,
        thresholds: np.ndarray = np.linspace(0, 1.0, 11),
    ) -> tuple[EvalMetrics, list[EvalMetrics]]:
        # calculate t2i and i2t scores
        i2t_sim, t2i_sim = self.run_retrieval()

        # obtain maximum obj size in each image
        loader = self.loader
        areas = np.array(
            [
                max(box_info.areas)
                for _, _, box_infos in self.loader
                for box_info in box_infos
            ],
            dtype=np.float32,
        )

        # calculate metrics
        return self.report_metrics(
            scores_i2t=i2t_sim.cpu().numpy(),
            scores_t2i=t2i_sim.cpu().numpy(),
            txt2img=loader.dataset.txt2img,
            img2txt=loader.dataset.img2txt,
            thresholds=thresholds,
            areas=areas,
        )

    def pretty_print(
        self, metrics: EvalMetrics, ret_str: bool = False
    ) -> None | str:
        result_str = "# Text to Image\n"
        result_str += "## R@K\n"
        result_str += f"R@1: {metrics.img_r1:.2f}, R@5: {metrics.img_r5:.2f}, R@10: {metrics.img_r10:.2f}\n\n"

        result_str += "# Image to Text\n"
        result_str += "## R@K\n"
        result_str += f"R@1: {metrics.txt_r1:.2f}, R@5: {metrics.txt_r5:.2f}, R@10: {metrics.txt_r10:.2f}\n\n"

        if ret_str:
            return result_str

        print(result_str)
        return None

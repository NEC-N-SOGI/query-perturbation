from abc import ABC, abstractmethod
from enum import Enum, auto

import torch

from query_perturbation.data.box_loader import BoxInfo


class BaseWeight(ABC):
    def __init__(self, num_ca_layer: int):
        self.num_ca_layer = num_ca_layer

    @abstractmethod
    def __call__(self, box_info: BoxInfo) -> torch.Tensor:
        """_summary_

        Args:
            box_info (BoxInfo): _description_

        Returns:
            Tensor: weights. shape = [#obj, #layer,]
        """
        pass


class AreaWeightType(Enum):
    NORM_AREA = auto()  # area
    NORM_AREA_INV = auto()  # 1 - area
    CENTERED_AREA = auto()  # area - 0.5
    CENTERED_AREA_INV = auto()  # 0.5 - area
    CONSTANT = auto()  # scale


class AreaBasedWeight(BaseWeight):
    def __init__(
        self,
        num_ca_layer: int,
        weight_scale: float = 1.0,
        weight_type: AreaWeightType = AreaWeightType.NORM_AREA,
    ):
        super().__init__(num_ca_layer)
        self.set_weight_type(weight_type)
        self.set_weight_scale(weight_scale)

    def set_weight_scale(self, weight_scale: float) -> None:
        self.weight_scale = weight_scale

    def set_weight_type(self, weight_type: AreaWeightType) -> None:
        self.weight_type = weight_type

        self.area_sign = (
            1
            if weight_type
            in [AreaWeightType.NORM_AREA, AreaWeightType.CENTERED_AREA]
            else -1
        )
        self.center = (
            0
            if weight_type
            in [AreaWeightType.NORM_AREA, AreaWeightType.NORM_AREA_INV]
            else 0.5
        )

    def __call__(self, box_info: BoxInfo) -> torch.Tensor:
        weight = torch.zeros(len(box_info.areas), self.num_ca_layer)
        for i, area in enumerate(box_info.areas):
            if self.weight_type == AreaWeightType.CONSTANT:
                weight[i, :] = self.weight_scale * torch.ones(self.num_ca_layer)
            else:
                weight[i, :] = self.weight_scale * (
                    self.area_sign * area + (-1 * self.area_sign) * self.center
                )

        return weight

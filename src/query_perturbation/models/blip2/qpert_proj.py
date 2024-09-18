import torch
from torch import nn
from torch.nn import Linear


class QueryProjection(nn.Module):
    def __init__(
        self,
        query: Linear,
        do_normalize: bool = False,
    ) -> None:
        super().__init__()
        self.query = query
        self.projection_mat: list[torch.Tensor] = []
        self.do_normalize = do_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.query(x)
        return x

    def set_projection_mat(self, projection_mat: list[torch.Tensor]) -> None:
        self.projection_mat = projection_mat

    def proj(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """

        Args:
            queries (_type_): [n_batch, n_head, n_query, n_dims]
            keys (_type_): [n_batch, n_head, n_keys, n_dims]

        Returns:
            _type_: _description_
        """
        assert (
            len(self.projection_mat) == queries.shape[0]
            or len(self.projection_mat) == 0
        ), f"#p_mats does not fit to bs. len(p_mats): {len(self.projection_mat)} != bs: {queries.shape[0]}"

        if queries.dim() == 3:
            # for each head
            new_x_shape = queries.size()[:-1] + (keys.shape[1], keys.shape[-1])
            _queries = queries.view(*new_x_shape).permute(0, 2, 1, 3)
        else:
            _queries = queries

        if len(self.projection_mat) > 0:
            if self.do_normalize:
                norm = torch.norm(_queries, dim=-1, keepdim=True)

            _queries_cp = _queries.clone()
            for i, (p, q) in enumerate(zip(self.projection_mat, _queries_cp)):
                if len(p) > 0:
                    if p.dim() > 3:
                        p = p.mean(0)  # some objs
                    projected = q @ p
                    projected = projected.masked_fill(torch.isnan(projected), 0)
                    _queries[i] = _queries[i] + projected

            if self.do_normalize:
                # keep the original norm
                _queries = norm * torch.nn.functional.normalize(_queries, dim=-1)

        if queries.dim() == 3:
            _queries = _queries.permute(0, 2, 1, 3).contiguous()

        return _queries.view(queries.shape).contiguous()

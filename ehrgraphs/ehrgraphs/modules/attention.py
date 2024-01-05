import torch


class PoolingMultiheadAttention(torch.nn.Module):
    """PMA block from Set Transformer: http://proceedings.mlr.press/v97/lee19d.html
    Short summary.

    Parameters
    ----------
    input_dim : int
        Number of features of set elements.
    output_dim : type
        Number of output / seed features.
    num_heads : type
        Number of heads for MultiheadAttention.
    """

    def __init__(self, input_dim: int, output_dim: int, num_heads: int):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.seed_vector = torch.nn.Parameter(torch.Tensor(1, 1, output_dim))
        torch.nn.init.xavier_uniform_(self.seed_vector)

        self.multihead_attn = torch.nn.MultiheadAttention(
            output_dim, num_heads, batch_first=False, kdim=input_dim, vdim=input_dim
        )

    def forward(self, embedding_set: torch.tensor, attn_mask: torch.tensor):
        inverse_mask = torch.logical_not(attn_mask)

        attn_output, attn_output_weights = self.multihead_attn(
            self.seed_vector.repeat(1, embedding_set.shape[0], 1),
            embedding_set.transpose(1, 0),
            embedding_set.transpose(1, 0),
            key_padding_mask=inverse_mask,
        )

        return torch.nan_to_num(attn_output[0])

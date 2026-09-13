from torch import Tensor
from torch import nn
import torch
import itertools
import numpy as np
from einops import rearrange
from typing import Dict, List, Tuple


def bin_times_to_frames(bin_times: List[float], frame_hz: int) -> List[int]:
    x = (torch.tensor(bin_times) * frame_hz).long().tolist()
    return x


class Codebook(nn.Module):
    def __init__(self, bin_frames):
        super().__init__()
        self.bin_frames = bin_frames
        self.n_bins: int = len(self.bin_frames)
        self.total_bins: int = self.n_bins * 2
        self.n_classes: int = 2 ** self.total_bins

        self.emb = nn.Embedding(
            num_embeddings=self.n_classes, embedding_dim=self.total_bins
        )
        self.emb.weight.data = self.create_code_vectors(self.total_bins) # just one-hot
        self.emb.weight.requires_grad_(False) # to make sure the codebook is not updated during training

    def single_idx_to_onehot(self, idx: int, d: int = 8) -> Tensor:
        assert idx < 2 ** d, "must be possible with {d} binary digits"
        z = torch.zeros(d)
        b = bin(idx).replace("0b", "")
        for i, v in enumerate(b[::-1]):
            z[i] = float(v)
        return z

    def create_code_vectors(self, n_bins: int) -> Tensor:
        """
        Create a matrix of all one-hot encodings representing a binary sequence of `self.total_bins` places
        Useful for usage in `nn.Embedding` like module.
        """
        n_codes = 2 ** n_bins
        embs = torch.zeros((n_codes, n_bins))
        for i in range(2 ** n_bins):
            embs[i] = self.single_idx_to_onehot(i, d=n_bins)
        return embs

    def encode(self, x: Tensor) -> Tensor:
        """

        Encodes projection_windows x (*, 2, 4) to indices in codebook (..., 1)

        Arguments:
            x:          Tensor (*, 2, 4)

        Inspiration for distance calculation:
            https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/vector_quantize_pytorch.py
        """
        assert x.shape[-2:] == (
            2,
            self.n_bins,
        ), f"Codebook expects (..., 2, {self.n_bins}) got {x.shape}"

        # compare with codebook and get closest idx
        shape = x.shape
        flatten = rearrange(x, "... c bpp -> (...) (c bpp)", c=2, bpp=self.n_bins)
        embed = self.emb.weight.T
        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        embed_ind = embed_ind.view(*shape[:-2])
        return embed_ind

    def decode(self, idx):
        v = self.emb(idx)
        v = rearrange(v, "... (c b) -> ... c b", c=2)
        return v

    def forward(self, projection_windows: Tensor):
        return self.encode(projection_windows)

# bin_times = [0.2, 0.4, 0.6, 0.8]
# bin_frames = bin_times_to_frames(bin_times, frame_hz=self.sr)
# self.codebook = Codebook(bin_frames=bin_frames)

class ObjectiveVAP(nn.Module):
    def __init__(
        self,
        bin_times: List[float] = [0.2, 0.4, 0.6, 0.8],
        frame_hz: int = 50,
        threshold_ratio: float = 0.5,
    ):
        super().__init__()
        self.frame_hz = frame_hz
        self.bin_times = bin_times
        self.bin_frames: List[int] = bin_times_to_frames(bin_times, frame_hz)
        self.horizon = sum(self.bin_frames)
        self.horizon_time = sum(bin_times)

        self.codebook = Codebook(self.bin_frames)
        self.requires_grad_(False)

    def __repr__(self):
        s = str(self.__class__.__name__)
        s += f"\n{self.codebook}"
        s += f"\n{self.projection_window_extractor}"
        s += "\n"
        return s

    @property
    def n_classes(self) -> int:
        return self.codebook.n_classes

    @property
    def n_bins(self) -> int:
        return self.codebook.n_bins

    def probs_next_speaker(
        self,
        probs: Tensor,
        from_bin: int = 0,
        to_bin: int = 3,
        scale_with_bins: bool = False,
    ) -> Tensor:
        assert (
            probs.ndim == 3
        ), f"Expected probs of shape (B, n_frames, n_classes) but got {probs.shape}"

        idx = torch.arange(self.codebook.n_classes).to(probs.device)
        states = self.codebook.decode(idx) # [256, 2, 4]

        if scale_with_bins:
            states = states * torch.tensor(self.bin_frames)
        abp = states[:, :, from_bin : to_bin + 1].sum(-1)  # sum speaker activity bins  # [256, 2]
        # Dot product over all states
        p_all = torch.einsum("bid,dc->bic", probs, abp) # bid * dc -> bic matrix, product of vap and vad to condition the vap on vad
        # probs shape: torch.Size([1, 18050, 256]), abp shape: torch.Size([256, 2])
        # => p_all is speaker activity at each timestep, conditioned on specific time range.
        # normalize
        p_all /= p_all.sum(-1, keepdim=True) + 1e-5 # sum to 1
        return p_all


class VAPDecoder:
    def __init__(self, bin_times: List[float]) -> None:
        self.bin_times = bin_times
        self.oj = ObjectiveVAP(bin_times=bin_times)

    def p_future(self, vap: Tensor) -> Tensor:
        if vap.ndim < 3:
            vap = vap.unsqueeze(0)
        return self.oj.probs_next_speaker(vap, 2, 3).squeeze()

    def p_now(self, vap: Tensor) -> Tensor:
        if vap.ndim < 3:
            vap = vap.unsqueeze(0)
        return self.oj.probs_next_speaker(vap, 0, 1).squeeze()

    def decode_probabilities(self, vap: Tensor) -> Tensor:
        if vap.ndim < 3:
            vap = vap.unsqueeze(0)
        p0 = self.oj.probs_next_speaker(vap, 0, 0)
        p1 = self.oj.probs_next_speaker(vap, 1, 1)
        p2 = self.oj.probs_next_speaker(vap, 2, 2)
        p3 = self.oj.probs_next_speaker(vap, 3, 3)
        return torch.stack((p0, p1, p2, p3), dim=-1).squeeze()

    def p_bc_idxs(self, who: int) -> Tensor:
        assert who in [0, 1]
        invert = who == 0
        combi = np.array(list(itertools.product([0, 1], repeat=3)))
        bottom = np.hstack((combi, np.ones((combi.shape[0], 1))))
        top = combi[1:, :]
        top = np.hstack((top, np.zeros((top.shape[0], 1))))
        result = []
        for i in range(top.shape[0]):
            for j in range(bottom.shape[0]):
                top_state, bottom_state = top[i], bottom[j]
                stack = (bottom_state, top_state) if invert else (top_state, bottom_state)
                result.append(np.stack(stack, axis=0))
        states = torch.tensor(np.stack(result, axis=0)).float()
        return self.oj.codebook.encode(states)

    def p_bc(self, vap: Tensor) -> Tensor:
        idx_0 = self.p_bc_idxs(0)
        idx_1 = self.p_bc_idxs(1)
        ap = vap[..., idx_0].sum(-1)
        bp = vap[..., idx_1].sum(-1)
        return torch.stack((ap, bp), dim=-1)

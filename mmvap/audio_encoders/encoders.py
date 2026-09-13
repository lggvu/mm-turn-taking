"""
Audio encoders.
"""
import torch
from torch import nn
import einops

from mmvap.audio_encoders.encoder_cpc_components import load_CPC, get_cnn_layer


class EncoderCPC(nn.Module):
    """
    Encoder: waveform -> h

    A simpler version of the Encoder used by the original (non-Lightning) codebase.
    """

    def __init__(self, load_pretrained=True, freeze=False):
        super().__init__()
        self.sample_rate = 16000
        self.encoder = load_CPC(load_pretrained)
        self.output_dim = self.encoder.gEncoder.conv4.out_channels
        self.dim = self.output_dim

        self.downsample = get_cnn_layer(
            dim=self.output_dim,
            kernel=[5],
            stride=[2],
            dilation=[1],
            activation="GELU",
        )  # temporal downsampler: 100hz -> 50hz
        self.downsample_ratio = 320

        if freeze:
            self.freeze()

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        print(f"Froze {self.__class__.__name__}!")

    def unfreeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        print(f"Trainable {self.__class__.__name__}!")

    def forward(self, waveform):
        if waveform.ndim < 3:
            waveform = waveform.unsqueeze(1)  # channel dim

        z = self.encoder.gEncoder(waveform)
        z = einops.rearrange(z, "b c n -> b n c")
        z = self.encoder.gAR(z)
        z = self.downsample(z)  # downsample by two times
        return z


class StereoEncoder(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = EncoderCPC()

    def forward(self, x):
        xl = self.encoder(x[:, :, 0])
        xr = self.encoder(x[:, :, 1])
        return torch.stack((xl, xr), dim=-1)

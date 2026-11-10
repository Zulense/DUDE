import torch 
from torch import nn
from einops import rearrange
import torch.nn.functional as F



class PEG3D(nn.Module):
    """
    Its job is to help the model understand local 3D neighborhoods (across time, height, and width) 
    while keeping computation lightweight, preventing the model from "seeing into the future," 
    and allowing long videos to be processed in smaller memory-friendly chunks without visual glitches.
    """
    def __init__(self, dim):
        super().__init__()
        self.ds_conv = nn.Conv3d(in_channels=dim,
                                 out_channels=dim,
                                 kernel_size=(3, 3, 3),
                                 groups=dim)
        self.is_first_chunk = True
        self.causal_cached = None
        self.tile_inference = False

    def forward(self, x):
        x = rearrange(x, 'b t h w d -> b d t h w')
        if self.tile_inference:
            if self.is_first_chunk:
                x = F.pad(x, (1, 1, 1, 1, 2, 0), value=0.)
            else:
                x = F.pad(x, (1, 1, 1, 1, 0, 0), value=0.)
                x = torch.concatenate((self.causal_cached, x), dim=2)

            self.causal_cached = x[:, :, -2:].clone()

        else:
            x = F.pad(x, (1, 1, 1, 1, 2, 0), value=0.)
        x = self.ds_conv(x.contiguous())
        x = rearrange(x, 'b d t h w -> b t h w d')
        return x


class GEGLU(nn.Module):
    """
    GEGLU (Gated Linear Unit with GELU) acts as a multiplicative filter 
    where one half of your feature channels `dictates` how much of the other half is allowed to `pass` through.
    """
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        # The network applies the non-linear Gaussian Error Linear Unit (GELU) exclusively to the gate tensor
        # The model actively learns to use the gate vector to scale, modulate, or suppress the signal in the x vector.
        return F.glu(gate) * x


def ffd(dim, mult=4, dropout=0.):
    """
    this block immediately follows the depthwise 3D convolution (PEG3D). While the convolution looks at neighboring pixels across space and time without mixing channels, 
    ffd does the exact opposite: it looks at one specific pixel and mixes all of its channels together to build higher-level representations.
    """
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False)
    )

class NAF(nn.Module):
    """
    The NAF (Neighborhood-Aware Feedforward) class acts as the assembly line that links the two components you just learned (PEG3D and ffd) into a single, cohesive module. 
    It applies a alternating "1-2 punch" to the video data, repeating this process however many times you specify.
    """
    def __init__(self,
                 num_layer,
                 dim):
        super().__init__()
        self.num_layer = num_layer
        self.deconv_layer = nn.Sequential()
        self.ffd_layer = nn.Sequential()
        for _ in range(num_layer):
            self.ffd_layer.append(ffd(dim, 4))
            self.deconv_layer.append(PEG3D(dim))

    def forward(self, x):
        for i in range(self.num_layer):
            x = self.deconv_layer[i](x)
            x = self.ffd_layer[i](x)
        return x




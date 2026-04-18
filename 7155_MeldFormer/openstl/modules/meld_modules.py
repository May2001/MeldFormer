import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, Mlp


class ST_PatchInflated(nn.Module):
    r""" Tensor to Patch Inflating

    Args:
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        input_resolution (tuple[int]): Input resulotion.
    """

    def __init__(self, in_chans, embed_dim, input_resolution, stride=8, padding=0, output_padding=0):
        super(ST_PatchInflated, self).__init__()

        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        output_padding = to_2tuple(output_padding)
        self.input_resolution = input_resolution

        self.Conv = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=stride,
                                       stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.Conv(x)

        return x




def get_1D_relative_position_index(T):
    coords = torch.arange(T)
    relative_coords = coords[:, None] - coords[None, :]
    relative_coords += T - 1
    return relative_coords

class LocalAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim=None, T=10, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.T=T
        table_size = 2 * T - 1
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self._1D_relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))

        # get pair-wise relative position index for each token inside the window
        self.register_buffer("_1D_relative_position_index", get_1D_relative_position_index(T))

        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self._1D_relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def _get_1D_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self._1D_relative_position_bias_table[
            self._1D_relative_position_index.view(-1)  
        ]


        relative_position_bias = relative_position_bias.view(self.T, self.T, -1)

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        return relative_position_bias.unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)

        """
        B_, N, C = x.shape
        assert N == self.T, f"input N: {N}!= time T: {self.T}"
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn + self._get_1D_rel_pos_bias()

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LocalTransformerBlock(nn.Module):
    def __init__(
            self, dim, num_heads=4, head_dim=None,T=10,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.T = T
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = LocalAttention(
            dim, num_heads=num_heads, head_dim=head_dim,T=T,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)


    def forward(self, x):
        B, L, C = x.shape
        assert L == self.T, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)

        x = self.attn(x)  # num_win*B, window_size*window_size, C


        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


def get_relative_position_index(global_h, global_w):

    coords = torch.stack(torch.meshgrid([torch.arange(global_h), torch.arange(global_w)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += global_h - 1
    relative_coords[:, :, 1] += global_w - 1
    relative_coords[:, :, 0] *= 2 * global_w - 1
    return relative_coords.sum(-1)

class GlobalAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim=None, global_size=16, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.global_size = to_2tuple(global_size)
        global_h,global_w=self.global_size
        self.global_area=global_h*global_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5


        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * global_h - 1) * (2 * global_w - 1), num_heads))


        self.register_buffer("relative_position_index", get_relative_position_index(global_h, global_w))

        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.global_area, self.global_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*T, N, C)

        """
        B_, N, C = x.shape
        #assert N == self.global_size**2, f"input N: {N}!= global_size: {self.T}**2"
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn + self._get_rel_pos_bias()

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GlobalTransformerBlock(nn.Module):
    def __init__(
            self, dim, input_resolution, num_heads=4, head_dim=None,global_size=16,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.global_size=global_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = GlobalAttention(
            dim, num_heads=num_heads, head_dim=head_dim,global_size=to_2tuple(self.global_size),
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)


    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H*W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)

        x = self.attn(x)


        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class LearnableCompressor(nn.Module):
    def __init__(self, input_dim=1280, output_dim=640):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.W, gain=1.0)

    def forward(self, x):
        return torch.matmul(x, self.W.t())  # z = xW^T

class Sensation(nn.Module):
    def __init__(self, in_chans=16):
        super().__init__()
        self.in_chans=in_chans
        self.convNorm3 = nn.Conv2d(in_channels=in_chans, out_channels=in_chans, kernel_size=3, stride=1, padding=1)
        self.convNorm5 = nn.Conv2d(in_channels=in_chans, out_channels=in_chans, kernel_size=5, stride=1, padding=2)
        self.convNorm7 = nn.Conv2d(in_channels=in_chans, out_channels=in_chans, kernel_size=7, stride=1, padding=3)
        self.convNorm11 = nn.Conv2d(in_channels=in_chans, out_channels=in_chans, kernel_size=11, stride=1, padding=5)
        if in_chans>1:
            self.norm = nn.GroupNorm(2, in_chans)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x3, x5, x7, x11 = torch.split(x, self.in_chans, dim=1)
        x3=self.convNorm3(x3)
        x5=self.convNorm5(x5)
        x7=self.convNorm7(x7)
        x11=self.convNorm11(x11)
        if self.in_chans>1:
            x3=self.norm(x3)
            x5=self.norm(x5)
            x7=self.norm(x7)
            x11=self.norm(x11)
        x3=self.act(x3)
        x5=self.act(x5)
        x7=self.act(x7)
        x11=self.act(x11)
        x = torch.cat([x3, x5, x7, x11], dim=1)
        return x
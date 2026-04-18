import torch
import torch.nn as nn


from openstl.modules import ST_PatchInflated,LocalTransformerBlock,GlobalTransformerBlock,LearnableCompressor,Sensation
from timm.models.layers import PatchEmbed,trunc_normal_


class Meld_Model(nn.Module):
    def __init__(self, in_shape, patch_size=2, embed_dim_local=256, embed_dim_global=64, num_heads=8,
            in_chans=16, local_n=3, global_n=6, spatial_n=3, mlp_ratio=4., qkv_bias=True, drop=0.,
            attn_drop=0., drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs ):
        super(Meld_Model, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        self.patch_embed = PatchEmbed(img_size=(H,W), patch_size=patch_size,
                                      in_chans=in_chans*4, embed_dim=embed_dim_local,
                                      norm_layer=norm_layer)
        patches_resolution = self.patch_embed.grid_size

        self.embed_dim_global = embed_dim_global
        self.embed_dim_local = embed_dim_local
        self.pos_drop = nn.Dropout(p=drop)
        self.patch_inflated = ST_PatchInflated( in_chans=in_chans*4 , embed_dim=embed_dim_local, stride= patch_size,
                                            input_resolution=patches_resolution)

        self.sensation = Sensation(in_chans= in_chans)
        self.readout=nn.Conv2d(in_chans*4,C,1)
        self.readin=nn.Conv2d(C,in_chans*4,1)
        self.compressor = LearnableCompressor(input_dim=embed_dim_local,output_dim=embed_dim_global)
        self.decompressor = LearnableCompressor(input_dim=embed_dim_global,output_dim=embed_dim_local)
        print("Meldmodel")

        enc_layers = []
        for _ in range(local_n):
            enc_layers.append(
                LocalTransformerBlock(dim=embed_dim_local, num_heads=num_heads, T=T, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      drop=drop,attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
            )
        self.enc = nn.Sequential(*enc_layers)
        hid_layers = []
        for _ in range(global_n):
            hid_layers.append(
                GlobalTransformerBlock(dim=embed_dim_global*T, input_resolution=(patches_resolution[0], patches_resolution[1]),
                                 num_heads=num_heads*T, global_size=(patches_resolution[0], patches_resolution[1]), mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path, norm_layer=norm_layer)
            )
        self.hid = nn.Sequential(*hid_layers)
        dec_layers = []
        for _ in range(local_n):
            dec_layers.append(
                LocalTransformerBlock(dim=embed_dim_local, num_heads=num_heads, T=T, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      drop=drop,attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
            )
        self.dec = nn.Sequential(*dec_layers)
        spatial_layers = []
        for _ in range(spatial_n):
            spatial_layers.append(
                GlobalTransformerBlock(dim=embed_dim_local, input_resolution=(patches_resolution[0], patches_resolution[1]),
                                 num_heads=num_heads*T, global_size=patches_resolution[0], mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path, norm_layer=norm_layer)
            )
        self.spatial = nn.Sequential(*spatial_layers)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)




    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        X = x_raw.view(B * T, C, H, W)

        x_=self.readin(X)
        x=self.sensation(x_)

        x = self.patch_embed(x)

        B_, L, C_ = x.shape
        x=x.view(B,T,L,C_).transpose(1, 2).contiguous().view(B*L,T,C_)
        x=self.enc(x)
        x=self.compressor(x)

        x=x.view(B,L,T*self.embed_dim_global )

        Y=self.hid(x)

        B_,L_,C_=Y.shape

        Y=Y.view(B_*L_,T,-1)
        Y=self.decompressor(Y)

        Y = self.dec(Y)

        Y=Y.view(B_,L_,T,-1).transpose(1, 2).contiguous().view(B_*T,L_,-1)
        Y=self.spatial(Y)
        Y_ = torch.sigmoid(self.patch_inflated(Y))

        Y=self.sensation(Y_)
        Y=self.readout(Y)
        Y = Y.reshape(B, T, C, H, W)
        return Y



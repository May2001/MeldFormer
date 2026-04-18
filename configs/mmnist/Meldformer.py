method = 'Meldformer'
# model
num_heads = 4
patch_size = 4
embed_dim_local=256
embed_dim_global=64
in_chans=8
local_n=3
global_n=6
spatial_n=0
# training
lr = 5e-4
batch_size = 16
sched = 'onecycle'
final_div_factor = 5e3


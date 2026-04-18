# Copyright (c) CAIRI AI Lab. All rights reserved

from .meldformer import Meldformer

method_maps = {
    
    'meldformer': Meldformer
}

__all__ = [
    'method_maps', 'ConvLSTM', 'E3DLSTM', 'MAU', 'MIM',
    'PredRNN', 'PredRNNpp', 'PredRNNv2', 'PhyDNet', 'SimVP', 'TAU',
    "MMVP", 'SwinLSTM_D', 'SwinLSTM_B', 'WaST', 'STformer','Meldformer'
]
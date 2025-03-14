
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from mamba_ssm.modules.mamba_simple import Mamba
from einops import rearrange, einsum
 
class MambaBlock(nn.Module):
    def __init__(self, dim=256, nlayer=4, dt_rank="auto", d_state=16, d_conv=4, expand=1):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            nn.Sequential(
                Mamba(
                    d_model=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                ),
                nn.LayerNorm(dim)
            ) for _ in range(nlayer)
        ])
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer[0](x)
            x = layer[1](x + residual)
            
        return self.proj(x)
class MambaEncoder(nn.Module):
    def __init__(self,  inc=256, outc=256, nlayer=4, d_state=16, d_conv=4, expand=1):
        super(MambaEncoder, self).__init__()
        self.mamba = MambaBlock(
            dim=inc,
            nlayer=nlayer,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank="auto"
        )
        self.norm_in = nn.InstanceNorm1d(outc)
        self.adaptor = nn.Sequential(
            nn.LayerNorm(outc),
            nn.Dropout(0.1)
        )

        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='linear')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, Mamba):
            nn.init.normal_(module.A_log, mean=0.0, std=0.01)
            if hasattr(module, 'out_proj'):
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0.01)
            for name, param in module.named_parameters():
                if 'dt_proj' in name:
                    nn.init.normal_(param, mean=0.0, std=0.02)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x = self.conv1(x)
        x = self.norm_in(x)
        # x = self.adaptor(x)
        out = self.mamba(x)

        
        return out

if __name__ == "__main__":
    batch = 16
    inc = 256
    seq_len = 300
    outc = 512

    dummy_input = torch.randn(batch, inc, seq_len).to("cuda")
    encoder = MambaEncoder(inc=inc, outc=outc, nlayer=4, d_state=16, d_conv=4, expand=2).to("cuda")

    output = encoder(dummy_input)
    print("Output shape:", output.shape)


    print("\n模型结构:")
    print(encoder)


    total_params = 0
    print("\n各层参数信息:")
    for name, param in encoder.named_parameters():
        num_params = param.numel()
        total_params += num_params
        print(f"{name}: {param.shape} ({num_params} parameters)")
    print(f"Total parameters: {total_params}")
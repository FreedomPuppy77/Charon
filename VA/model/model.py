import torch
import torch.nn as nn
import yaml
from munch import DefaultMunch
from torch.utils.tensorboard import SummaryWriter
from .tcn import TemporalConvNet
from .mamba import MambaEncoder


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        tcn_channels = [1024, 768, 512, 256]
        self.temporal = TemporalConvNet(
            num_inputs=tcn_channels[0],
            num_channels=tcn_channels,
            kernel_size=cfg.Model.kernel_size,
            dropout=cfg.HP.dropout,
            attention=False,
        )
        self.mamba = MambaEncoder(
            inc=tcn_channels[-1],
            outc=cfg.Model.out_dim,
            nlayer=cfg.Model.num_layer,
            d_state=cfg.Model.d_state,
            d_conv=cfg.Model.d_conv,
            expand=cfg.Model.expand,
        )
        self.dual_reg = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        if x.shape[2] == 1:
            x = x[:, :, 0, :]
        bs, seq_len, _ = x.shape
        x = x.transpose(1, 2) # [B, D, L]
        x = self.temporal(x)
        x = self.mamba(x)
        predictions = self.dual_reg(x)
        return {
            'valence': predictions[..., 0],
            'arousal': predictions[..., 1]
        }


if __name__ == "__main__":
    config_path = "VA/config/config_fold1.yml"
    yaml_dict = yaml.load(
        open(config_path, "r", encoding="utf-8"), Loader=yaml.FullLoader
    )
    cfg = DefaultMunch.fromDict(yaml_dict)
    model = Model(cfg)
    print(model)
    writer = SummaryWriter(log_dir="runs/model_visualization")
    dummy_input = torch.randn(16, 300, 1024)
    writer.add_graph(model, dummy_input)
    writer.close()



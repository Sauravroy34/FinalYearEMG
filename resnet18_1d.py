"""1-D ResNet-18 backbone: encodes one EMG window (C, L) into a feature vector."""
import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet18_1D(nn.Module):
    """Standard ResNet-18 layout ([2, 2, 2, 2] basic blocks) with 1-D convolutions.

    Input : (B, in_channels, L)   e.g. (B, 1, 500) for a 500 ms EMG window @ 1 kHz
    Output: (B, out_dim)          out_dim = 8 * base_width (512 by default)
    Global average pooling makes it length agnostic (250 ms pre-training windows
    and 500 ms fine-tuning windows can share weights).
    """

    def __init__(self, in_channels: int = 1, base_width: int = 64):
        super().__init__()
        w = base_width
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, w, 7, 2, 3, bias=False),
            nn.BatchNorm1d(w),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1),
        )
        self.layer1 = self._make_layer(w, w, 2, 1)
        self.layer2 = self._make_layer(w, 2 * w, 2, 2)
        self.layer3 = self._make_layer(2 * w, 4 * w, 2, 2)
        self.layer4 = self._make_layer(4 * w, 8 * w, 2, 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = 8 * w
        self._init_weights()

    @staticmethod
    def _make_layer(in_ch, out_ch, blocks, stride):
        layers = [BasicBlock1D(in_ch, out_ch, stride)]
        layers += [BasicBlock1D(out_ch, out_ch, 1) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.pool(x).flatten(1)


if __name__ == "__main__":
    m = ResNet18_1D()
    print(m(torch.randn(4, 1, 500)).shape, sum(p.numel() for p in m.parameters()) / 1e6, "M params")

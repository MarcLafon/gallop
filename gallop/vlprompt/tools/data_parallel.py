import torch.nn as nn


class DataParallel(nn.DataParallel):

    @property
    def width(self) -> int:
        return self.module.width

    @property
    def layers(self) -> int:
        return self.module.layers

    @property
    def resblocks(self) -> nn.Sequential:
        return self.module.resblocks

    @property
    def conv1(self) -> nn.Conv2d:
        return self.module.conv1

    @property
    def proj(self) -> nn.Linear:
        return self.module.proj

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):

    def __init__(self, in_chans=1, depths=[3, 3, 27, 3], dims=[32,64,64,128],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3],
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=1, stride=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out =self.shared_MLP(self.avg_pool(x))
        max_out =self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class SeparableConv2d_same(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1,padding=0,dilation=1, bias=False):
        super(SeparableConv2d_same, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding=padding,dilation=dilation, groups=inplanes, bias=bias)
        self.pointwise1 = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)
        self.bn1=nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)


        self.conv2 = nn.Conv2d(planes, planes, kernel_size, stride, padding=padding, dilation=dilation,groups=planes, bias=bias)
        self.pointwise2 = nn.Conv2d(planes, planes, 1, 1, 0, 1, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.pointwise2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class EMV_Net(nn.Module):
    def __init__(self):
        super(EMV_Net, self).__init__()
        self.ConvNeXt = ConvNeXt()
        self.up4 =nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, dilation=1)
        self.up3 =nn.ConvTranspose2d(96, 96, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, dilation=1)
        self.up2 =nn.ConvTranspose2d(80, 80, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, dilation=1)

        self.conv0_4 = SeparableConv2d_same(192, 96, 3, 1, 1, 1)
        self.conv0_3 = SeparableConv2d_same(160, 80, 3, 1, 1, 1)
        self.conv0_2 = SeparableConv2d_same(112, 56, 3, 1, 1, 1)
        self.conv0_1 = SeparableConv2d_same(56, 9, 3, 1, 1, 1)

        self.cbam4 = CBAM(192)
        self.cbam3 = CBAM(160)
        self.cbam2 = CBAM(112)

    def forward(self, input):
        layer1,layer2,layer3,layer4=self.ConvNeXt(input)

        layer4 = self.up4(layer4)
        layer4 = torch.cat((layer3, layer4), dim=1)
        layer4=self.cbam4(layer4)
        layer4 = self.conv0_4(layer4)

        layer3 = self.up3(layer4)
        layer3 = torch.cat((layer3, layer2), dim=1)
        layer3=self.cbam3(layer3)
        layer3 = self.conv0_3(layer3)

        layer2 = self.up2(layer3)
        layer2 = torch.cat((layer1, layer2), dim=1)
        layer2=self.cbam2(layer2)
        layer2 = self.conv0_2(layer2)

        layer1 = self.conv0_1(layer2)

        return layer1


if __name__ == "__main__":
    model = EMV_Net()
    model.eval()
    image = torch.randn(2, 1, 480, 736)
    out = model(image)
    print(out.size())

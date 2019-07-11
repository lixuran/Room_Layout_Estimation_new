import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.model_zoo as model_zoo

from DCN_local.modules.deform_conv import DeformConvPack
from DCN_local.modules.modulated_deform_conv import ModulatedDeformConv


def lr_pad(x, padding=1):
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=3)


class LR_PAD(nn.Module):
    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return lr_pad(x, self.padding)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=(1, 0), bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(lr_pad(x, 1))
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(lr_pad(out, 1))
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(lr_pad(out))
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class EquiBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(EquiBottleneck, self).__init__()
        self.conv1 = DeformConvPack(inplanes, planes, kernel_size=1,stride=1,padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DeformConvPack(planes, planes, kernel_size=3, stride=stride,
                               padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = DeformConvPack(planes, planes * self.expansion, kernel_size=1,stride=1,padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(lr_pad(out))
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class EquiBottleneck_semi(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(EquiBottleneck_semi, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DeformConvPack(planes, planes, kernel_size=3, stride=stride,
                               padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(lr_pad(out))
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=(3, 0),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x):
        conv_list = []
        x = self.conv1(lr_pad(x, 3))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x); conv_list.append(x)
        x = self.layer2(x); conv_list.append(x)
        x = self.layer3(x); conv_list.append(x)
        x = self.layer4(x); conv_list.append(x)

        return conv_list




def resnet50(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model
    
def equiresnet50(pretrained=False, **kwargs):
    model = ResNet(EquiBottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model
    
def semi_equiresnet50(pretrained=False, **kwargs):
    model = ResNet(EquiBottleneck_semi, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model
    
def conv3x3(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True))

        
class VAE(nn.Module):
    def __init__(self, backbone, use_rnn):
        super(HorizonNet, self).__init__()
        self.backbone = backbone
        self.use_rnn = use_rnn
        if backbone == 'resnet18':
            self.feature_extractor = resnet18()
            _exp = 1
        elif backbone == 'resnet50':
            self.feature_extractor = resnet50()
            _exp = 4
        elif backbone == 'resnet101':
            self.feature_extractor = resnet101()
            _exp = 4
        elif backbone =='equiresnet50':
            self.feature_extractor = equiresnet50()
            _exp = 4
        elif backbone =='semi_equiresnet50':
            self.feature_extractor = semi_equiresnet50()
            _exp = 4  
              
        else:
            raise NotImplementedError()
    def forward(self,x):
        
        
class Decoder(nn.Module):
    def __init__(self, skip_num=2, out_planes=3):
        super(Decoder, self).__init__()
        self.convs = nn.ModuleList([
            conv3x3(2048, 1024),
            conv3x3(1024 * skip_num, 512),
            conv3x3(512 * skip_num, 256),
            conv3x3(256 * skip_num, 128),
            conv3x3(128 * skip_num, 64),
            conv3x3(64 * skip_num, 32)])
        self.last_conv = nn.Conv2d(
            32 * skip_num, out_planes, kernel_size=3, padding=1)

    def forward(self, f_list):
        conv_out = []
        f_last = f_list[0]
        for conv, f in zip(self.convs, f_list[1:]):
            f_last = F.interpolate(f_last, scale_factor=2, mode='nearest')
            f_last = conv(f_last)
            f_last = torch.cat([f_last, f], dim=1)
            conv_out.append(f_last)
        conv_out.append(self.last_conv(F.interpolate(
            f_last, scale_factor=2, mode='nearest')))
        return conv_out


if __name__ == '__main__':

    encoder = Encoder()
    edg_decoder = Decoder(skip_num=2, out_planes=3)
    cor_decoder = Decoder(skip_num=3, out_planes=1)

    with torch.no_grad():
        x = torch.rand(2, 6, 512, 1024)
        en_list = encoder(x)
        edg_de_list = edg_decoder(en_list[::-1])
        cor_de_list = cor_decoder(en_list[-1:] + edg_de_list[:-1])

    for f in en_list:
        print('encoder', f.size())
    for f in edg_de_list:
        print('edg_decoder', f.size())
    for f in cor_de_list:
        print('cor_decoder', f.size())

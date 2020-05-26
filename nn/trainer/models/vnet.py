import torch
import torch.nn as nn
import torch.nn.functional as F

class VNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        '''
        ConvBlock = consistent convs
        for each conv, conv(5x5) -> BN -> activation(PReLU)
        params:
        in/out channels: output/input channels
        layers: number of convolution layers
        '''
        super(VNetConvBlock, self).__init__()
        self.layers = layers
        self.afs = torch.nn.ModuleList() # activation functions
        self.convs = torch.nn.ModuleList() # convolutions
        self.bns = torch.nn.ModuleList()
        # first conv
        self.convs.append(nn.Conv2d( \
                in_channels, out_channels, kernel_size=5, padding=2))
        self.bns.append(nn.BatchNorm2d(out_channels))
        self.afs.append(nn.PReLU(out_channels))
        #self.afs.append(nn.ELU())
        for i in range(self.layers-1):
            self.convs.append(nn.Conv2d( \
                    out_channels, out_channels, kernel_size=5, padding=2))
            self.bns.append(nn.BatchNorm2d(out_channels))
            self.afs.append(nn.PReLU(out_channels))

    def forward(self, x):
        out = x
        for i in range(self.layers):
            out = self.convs[i](out)
            out = self.bns[i](out)
            out = self.afs[i](out)
        return out

class VNetInBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers=1):
        super(VNetInBlock, self).__init__()
        #self.bn = nn.BatchNorm2d(in_channels)
        self.convb = VNetConvBlock(in_channels, out_channels, layers=layers)

    def forward(self, x):
        #out = self.bn(x)
        out = self.convb(x)
        out = torch.add(out, x)
        return out

class VNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super(VNetDownBlock, self).__init__()
        self.down = nn.Conv2d( \
                in_channels, out_channels, kernel_size=2, stride=2)
        self.af= nn.PReLU(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.convb = VNetConvBlock(out_channels, out_channels, layers=layers)

    def forward(self, x):
        down = self.down(x)
        down = self.bn(down)
        down = self.af(down)
        out = self.convb(down)
        out = torch.add(out, down)
        return out

class VNetUpBlock(nn.Module):
    def __init__(self, in_channels, br_channels, out_channels, layers):
        super(VNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(\
                in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.af= nn.PReLU(out_channels)
        self.convb = VNetConvBlock( \
                out_channels+br_channels, out_channels, layers=layers)

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.bn(up)
        up = self.af(up)
        out = torch.cat([up, bridge], 1)
        out = self.convb(out)
        out = torch.add(out, up)
        return out

class VNetOutBlock(nn.Module):
    def __init__(self, \
            in_channels, br_channels, out_channels, classes, layers=1):
        super(VNetOutBlock, self).__init__()
        self.up = nn.ConvTranspose2d(\
                in_channels, out_channels, kernel_size=2, stride=2)
        self.bn_up = nn.BatchNorm2d(out_channels)
        self.af_up= nn.PReLU(out_channels)
        self.convb = VNetConvBlock( \
                out_channels+br_channels, out_channels, layers=layers)
        self.conv = nn.Conv2d(out_channels, classes, kernel_size=1)
        self.bn_out = nn.BatchNorm2d(classes)
        self.af_out= nn.PReLU(classes)

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.bn_up(up)
        up = self.af_up(up)
        out = torch.cat([up, bridge], 1)
        out = self.convb(out)
        out = torch.add(out, up)
        out = self.conv(out)
        out = self.bn_out(out)
        out = self.af_out(out)
        return out

class VNet(nn.Module):
    def __init__(self, classes_num = 2, \
            inBlock=VNetInBlock, downBlock=VNetDownBlock, \
            upBlock=VNetUpBlock, outBlock=VNetOutBlock):
        classes = classes_num
        super(VNet, self).__init__()
        self.in_block = inBlock(1, 16, 1)
        self.down_block1 = downBlock(16, 32, 2)
        self.down_block2 = downBlock(32, 64, 3)
        self.down_block3 = downBlock(64, 128, 3)
        self.down_block4 = downBlock(128, 256, 3)
        self.up_block1 = upBlock(256, 128, 256, 3)
        self.up_block2 = upBlock(256, 64, 128, 3)
        self.up_block3 = upBlock(128, 32, 64, 2)
        self.out_block = outBlock(64, 16, 32, classes, 1)

    def forward(self, x):
        br1 = self.in_block(x)
        br2 = self.down_block1(br1)
        br3 = self.down_block2(br2)
        br4 = self.down_block3(br3)
        out = self.down_block4(br4)
        out = self.up_block1(out, br4)
        out = self.up_block2(out, br3)
        out = self.up_block3(out, br2)
        out = self.out_block(out, br1)
        return out

class VNet7(nn.Module):
    def __init__(self, classes_num = 2):
        classes = classes_num
        super(VNet7, self).__init__()
        self.encoder1 = VNetInBlock(1, 16, 1)
        self.encoder2 = VNetDownBlock(16, 32, 2)
        self.encoder3 = VNetDownBlock(32, 64, 3)
        self.encoder4 = VNetDownBlock(64, 128, 3)
        self.encoder5 = VNetDownBlock(128, 256, 3)
        self.encoder6 = VNetDownBlock(256, 256, 3)
        self.bottleneck = VNetDownBlock(256, 256, 3)
        self.decoder6 = VNetUpBlock(256, 256, 256, 3)
        self.decoder5 = VNetUpBlock(256, 256, 256, 3)
        self.decoder4 = VNetUpBlock(256, 128, 256, 3)
        self.decoder3 = VNetUpBlock(256, 64, 128, 3)
        self.decoder2 = VNetUpBlock(128, 32, 64, 2)
        self.decoder1 = VNetOutBlock(64, 16, 32, classes, 1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        b = self.encoder6(e6)
        d6 = self.decoder6(b, e6)
        d5 = self.decoder5(d6, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        return d1

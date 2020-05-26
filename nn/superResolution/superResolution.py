import sys
import os
import torch
import numpy as np
import SimpleITK as sitk
import skimage.transform

class ResModule(torch.nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.subnet = torch.nn.Sequential(*modules)

    def forward(self, x):
        return x + self.subnet(x)

class ResNet(torch.nn.Module):
    def __init__(self, ch_in, ch=None):
        super().__init__()
        if ch is None:
            ch = ch_in
        self.convs = torch.nn.Sequential( \
                torch.nn.Conv3d(ch_in, ch, kernel_size=3, padding=1),
                torch.nn.LeakyReLU(0.2, True),
                torch.nn.Conv3d(ch, ch, kernel_size=3, padding=1))
        self.act = torch.nn.LeakyReLU(0.2, True)
        if ch_in != ch:
            self.shortcut = torch.nn.Conv3d(ch_in, ch, 1)
        else:
            self.shortcut = torch.nn.Sequential()

    def forward(self, x):
        return self.act(self.shortcut(x) + self.convs(x))

def Fine():
    ch = 48
    input_ch = 1
    output_ch = 1
    return ResModule(torch.nn.Conv3d(input_ch, ch, 1), \
            ResNet(ch), ResNet(ch), ResNet(ch), ResNet(ch), \
            ResNet(ch), ResNet(ch), ResNet(ch), ResNet(ch), \
            ResNet(ch), ResNet(ch), ResNet(ch), ResNet(ch), \
            torch.nn.Conv3d(ch, output_ch, 1))


class superResolution(object):
    def __init__(self):
        super(superResolution, self).__init__()
        self.model = Fine().cuda()
        self.model.load_state_dict(torch.load( \
                os.path.join(os.path.dirname(__file__), 'latest_net_G.pth')))

    def __call__(self, *args, **kw):
        return self.generate(*args, **kw)

    def generate(self, image):
        quantile = 5/1000.0
        targetSpacing = np.array([0.3906,0.3906])
        targetShape = np.array((256,256))
        spacing = np.array(image.GetSpacing()[::-1])[1:]
        model = self.model.eval()
        data = sitk.GetArrayFromImage(image)
        data = data[:,:,::-1]
        minmax = np.quantile(data.flatten(), (quantile, 1-quantile))
        data = (data - minmax[0])/(minmax[1]-minmax[0])
        out = []
        for i in range(data.shape[0]):
            scaled = skimage.transform.rescale( \
                    data[i,:,:], spacing/targetSpacing, order=1, \
                    anti_aliasing=False, mode='reflect', multichannel=False)
            scaled = np.pad(scaled, targetShape.max(), 'reflect')
            cropstart = ((scaled.shape - targetShape)/2.0).astype(np.int)
            cropstop = targetShape + cropstart
            cropped = scaled[cropstart[0]:cropstop[0], cropstart[1]:cropstop[1]]
            out.append(cropped.astype(np.float32))
        out = np.array(out)
        outsize = list(out.shape)
        outsize[0] = (outsize[0]-1)*4+1
        out = torch.Tensor(out).cuda().unsqueeze(0).unsqueeze(0)
        out = torch.nn.functional.interpolate( \
                out, size=outsize, mode='trilinear', align_corners=True)
        with torch.no_grad():
            out = model(out)[0][0].cpu().numpy()
        out = out*(minmax[1]-minmax[0]) + minmax[0]
        out = out[:,:,::-1]
        out = sitk.GetImageFromArray(out)
        sitk.WriteImage(out, 'test_SRout.nii')
        #out.CopyInformation(image)
        return out

if __name__ == '__main__':
    i = superResolution() 

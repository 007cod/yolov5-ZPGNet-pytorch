from collections import OrderedDict

import torch
import torch.nn as nn
from nets.conv_block import ConvBNReLU

def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


class ZPG_MA(nn.Module):
    def __init__(self, out_filters, MA_k=2,):
        super(ZPG_MA, self).__init__()
        self.MP = nn.ModuleList([
            nn.MaxPool2d(7, stride=4, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.MaxPool2d(3, stride=1, padding=1)
        ])
        self.MA_k = MA_k

        self.up = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Upsample(scale_factor=2, mode='nearest')
        ])
        self.CM = nn.ModuleList([
            CM(out_filters[-k],8) for k in range(1,MA_k+1)
        ])

    def forward(self, x, z):
        z = [self.MP[k](z) for k in range(3)]

        for i in range(self.MA_k):
            if i==0:
                m = z[0]
            else:
                m = z[i]+self.up[i](z[i-1])
            x[i],z[i] = self.CM[i](x[i],m)

        return x[0], x[1], x[2]
    
class ZPG_MA_BE(nn.Module):
    def __init__(self, out_filters, MA_k=2,):
        super(ZPG_MA_BE, self).__init__()
        self.MP = nn.ModuleList([
            nn.MaxPool2d(7, stride=4, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.MaxPool2d(3, stride=1, padding=1)
        ])
        self.MA_k = MA_k

        
        self.up = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Upsample(scale_factor=2, mode='nearest')
        ])
        self.CM = nn.ModuleList([
            CM(out_filters[-k],8) for k in range(1,MA_k+1)
        ])

        self.B1_CBR1 = ConvBNReLU(out_filters[-1], out_filters[-2], 3, 1, 1)
        self.B1_UP = nn.Upsample(scale_factor=2, mode='nearest')
        self.B1_CBR2 = ConvBNReLU(out_filters[-2] * 2, out_filters[-2], 3, 1, 1)

        self.B2_CBR1 = ConvBNReLU(out_filters[-2], out_filters[-3], 3, 1, 1)
        self.B2_UP = nn.Upsample(scale_factor=2, mode='nearest')
        self.B2_CBR2 = ConvBNReLU(out_filters[-3] * 2, out_filters[-3], 3, 1, 1)

        self.NE = NE(out_filters[-4], out_filters[-4] // 2)

        self.L_C3 = ConvBNReLU(out_filters[-4], out_filters[-3], 3, 2, 1)
        self.L_C2 = ConvBNReLU(out_filters[-3], out_filters[-2], 3, 2, 1)
        self.L_C1 = ConvBNReLU(out_filters[-2], out_filters[-1], 3, 2, 1)

        self.L_CBR2 = ConvBNReLU(out_filters[-3] * 3, out_filters[-3], 3, 1, 1)
        self.L_CBR1 = ConvBNReLU(out_filters[-2] * 3, out_filters[-2], 3, 1, 1)
        self.L_CBR0 = ConvBNReLU(out_filters[-1] * 2, out_filters[-1], 3, 1, 1)

    def forward(self, x, z):
        z = [self.MP[k](z) for k in range(3)]

        for i in range(self.MA_k):
            if i==0:
                m = z[0]
            else:
                m = z[i]+self.up[i](z[i-1])
            x[i],z[i] = self.CM[i](x[i],m)

        Lx3_in = self.L_C3(self.NE(x[3]))

        bx0 = self.B1_UP(self.B1_CBR1(x[0]))
        bx1_in = torch.cat([bx0, x[1]], dim=1)
        bx1_in = self.B1_CBR2(bx1_in)

        bx1 = self.B2_UP(self.B2_CBR1(x[1]))
        bx2_in = torch.cat([bx1, x[2]], dim=1)
        bx2_in = self.B2_CBR2(bx2_in)

        x2_out = self.L_CBR2(torch.cat([Lx3_in, bx2_in, x[2]], dim=1))
        Lx2_in = self.L_C2(x2_out)

        x1_out = self.L_CBR1(torch.cat([Lx2_in, bx1_in, x[1]], dim=1))
        Lx1_in = self.L_C1(x1_out)

        x0_out = self.L_CBR0(torch.cat([Lx1_in, x[0]], dim=1))

        return x0_out, x1_out, x2_out

class NE(nn.Module):
    def __init__(self, in_c, hidden_dim, k=(3,5,7)):
        super(NE, self).__init__()
        self.con1 = conv2d(in_c, hidden_dim, 3)
        self.mp = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(kernel_size=x, stride=2, padding=(x-1) // 2),
                nn.Upsample(scale_factor=2, mode='nearest')
            )
            for x in k
        ])
        self.CBRS = nn.ModuleList([
            ConvBNReLU(hidden_dim, hidden_dim, 3,1,1) for x in k
        ])
        self.con2 = conv2d(hidden_dim, in_c, 3)

    def forward(self, x):
        x = self.con1(x)
        for i in range(len(self.mp)):
            up = self.mp[i](x)
            x = self.CBRS[i](up + x)

        out = self.con2(x)
        return out
class SENet(nn.Module):
    def __init__(self, in_planes, ration = 16):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 通道数不变，H*W变为1*1
        self.fc1 = nn.Conv2d(in_planes , in_planes // 16 , 1 , bias = False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes//16 , in_planes ,1, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x * self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        return self.sigmoid(out)


class CM(nn.Module):
    def __init__(self, in_c, z_c):
        super(CM, self).__init__()
        self.in_c = in_c
        self.z_c = z_c
        self.SENet = SENet(in_c + z_c)
        self.CBR1 = ConvBNReLU(in_c + z_c, in_c + z_c, 3, 1, 1)
        self.CBR2 = ConvBNReLU(in_c, in_c, 3, 1, 1)

    def forward(self, x, z):
        out = torch.cat([x, z], dim=1)
        out = self.SENet(out)
        out = self.CBR1(out)
        x, z = out.split([self.in_c, self.z_c], dim=1)
        x = self.CBR2(x)
        return x, z

class ZPG(nn.Module):
    def __init__(self, alpha=0.4, beta=0.5,k=(3,7)):
        super(ZPG, self).__init__()
        self.process_layers = nn.Sequential(
            ConvBNReLU(1, 4, 7, 2, 3),
            ConvBNReLU(4, 2, 1),
            ConvBNReLU(2, 8, 3, stride=2, padding=1),
            ConvBNReLU(8, 4, 1),
            ConvBNReLU(4, 16, 3, stride=2, padding=1),
            ConvBNReLU(16, 8, 1),
        )
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        x = self.get_z_img(x)
        x = self.process_layers(x)
        return x

    def get_z_img(self, x):
        '''
        :param img: BGR img, shape = (b,3,h,w)
        :return: z_img, shape = (b,1,h,w)
        '''
        img = x.permute((0, 3, 2, 1))
        max_id = torch.argmax(img, dim=-1)
        mg = torch.sum(img, dim=-1)
        mask = torch.where(mg == 255 * 3, 0, 1).unsqueeze(-1)
        base_value = max_id + self.alpha
        b, gi, gj, _ = img.shape

        max_id = max_id.unsqueeze(-1)
        max_value = torch.gather(img, -1, max_id)

        other_value = torch.sum(img, dim=-1).unsqueeze(-1) - max_value
        other_value = other_value * (1 - self.alpha) * (1 - self.beta) / 510

        max_value = max_value * (1 - self.alpha) * self.beta / 255
        z_img = base_value.unsqueeze(-1) + max_value + other_value
        z_img = z_img * mask / 3
        z_img = z_img.permute((0, 3, 2, 1))

        return z_img

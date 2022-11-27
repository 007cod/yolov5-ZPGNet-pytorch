import torch
import torch.nn as nn

class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size,padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg,max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return out


class DM2(nn.Module):
    def __init__(self, high_c, low_c):
        super(DM2, self).__init__()
        self.SAM1 = SAM(3)
        self.SAM2 = SAM(3)
        self.conv = nn.Conv2d(high_c, low_c, 1)

    def forward(self, high_f, low_f):
        high_f = self.conv(high_f)
        ca1 = self.SAM1(high_f)

        ca2 = self.SAM2(low_f + high_f)

        da = ca2 - ca1

        return low_f * da


if __name__ == '__main__':
    a = torch.zeros((2,3,4,4))
    b = torch.zeros((2,6,4,4))
    DM = DM2(6,3)
    ans = DM(b,a)
    print(ans)

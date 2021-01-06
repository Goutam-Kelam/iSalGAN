import torch
import torch.nn.functional as F
from torch import nn

from resnext.resnext101 import ResNeXt101 #as ResNeXt101


class iSalGan(nn.Module):
    def __init__(self):
        super(iSalGan, self).__init__()
        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.reduce_low = nn.Sequential(
            nn.Conv2d(64 + 256 + 512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce_high = nn.Sequential(
            nn.Conv2d(1024 + 2048, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            _ASPP(256)
        )
        self.predict_feature = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.predict_sal = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )


        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True


    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        l0_size = layer0.size()[2:]

        reduce_low = self.reduce_low(torch.cat((
            layer0,
            F.upsample(layer1, size=l0_size, mode='bilinear', align_corners=True),
            F.upsample(layer2, size=l0_size, mode='bilinear', align_corners=True)), 1))

        reduce_high = self.reduce_high(torch.cat((
            layer3,
            F.upsample(layer4, size=layer3.size()[2:], mode='bilinear', align_corners=True)), 1))

        reduce_high = F.upsample(reduce_high, size=l0_size, mode='bilinear', align_corners=True)


        #print(reduce_low.size())
        #print(reduce_high.size())

        predict1 = self.predict_feature(reduce_high)
        predict2 = self.predict_feature(reduce_low)

        #print(predict1.size())
        #print(predict2.size())

        predict1 = F.upsample(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.upsample(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)

        #print(predict1.size())
        #print(predict2.size())

        high_sal_map = F.sigmoid(predict1)
        low_sal_map = F.sigmoid(predict2)

        #print(high_sal_map.size())
        #print(low_sal_map.size())


        concat_sal = self.predict_sal(torch.cat((predict1,predict2), 1))
        #print(concat_sal.size())
        concat_sal = F.sigmoid(F.upsample(concat_sal, size=x.size()[2:], mode='bilinear', align_corners=True))
        #print(concat_sal.size())
        #assert False
        return (high_sal_map, low_sal_map, concat_sal)



class _ASPP(nn.Module):
    #  this module is proposed in deeplabv3 and we use it in all of our baselines
    def __init__(self, in_dim):
        super(_ASPP, self).__init__()
        down_dim = in_dim // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear',
                           align_corners=True)
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))

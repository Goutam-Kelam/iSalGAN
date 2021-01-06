import numpy as np
import os

import torch
from PIL import Image
from torchvision import transforms

from config import msra10k_path
#from config import ecssd_path, hkuis_path, pascals_path, sod_path, dutomron_path
from misc import check_mkdir, crf_refine, AvgMeter, cal_precision_recall_mae, cal_fmeasure
#from model import R3Net
from iSalGan_generator import iSalGan

torch.manual_seed(2019)

# set which gpu to use
torch.cuda.set_device(0)

# the following two args specify the location of the file of trained model (pth extension)
# you should have the pth file in the folder './$ckpt_path$/$exp_name$'
ckpt_path = './ckpt'
exp_name = 'iSalGan'

args = {
    'snapshot': '6000',  # your snapshot filename (exclude extension name)
    'crf_refine': True,  # whether to use crf to refine results
    'save_results': True  # whether to save the resulting masks
}

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
to_pil = transforms.ToPILImage()

#to_test = {'ecssd': ecssd_path, 'hkuis': hkuis_path, 'pascal': pascals_path, 'sod': sod_path, 'dutomron': dutomron_path}

to_test = {'msra10k': msra10k_path}

def main():
    net = iSalGan().cuda()

    print('load snapshot \'%s\' for testing' % args['snapshot'])
    net.load_state_dict(torch.load("/home/gautam/Project/iSalGan/ckpt/iSalGan/%generator.pth"))
    net.eval()

    img = Image.open("/home/gautam/Project/iSalGan/test/My_pic.jpg").convert("RGB")
    img = img_transform(img).unsqueeze(0).cuda()
    predict = net(img)
    im1 = to_pil(predict[0].data.squeeze(0).cpu())
    im2 = to_pil(predict[1].data.squeeze(0).cpu())
    im3 = to_pil(predict[2].data.squeeze(0).cpu())

    im1.save("/home/gautam/Project/iSalGan/test/high_saliency_map.jpg")
    im2.save("/home/gautam/Project/iSalGan/test/low_saliency_map.jpg")
    im3.save("/home/gautam/Project/iSalGan/test/iSalGan_map.jpg")


if __name__ == '__main__':
    main()

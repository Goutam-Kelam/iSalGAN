import datetime
import os
import time

import torch
from torch import nn
from torch import optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import transforms

import joint_transforms
from config import msra10k_path
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir

from iSalGan_generator import iSalGan
from iSalGan_discriminator import Discriminator

from torch.backends import cudnn

from tqdm import tqdm
from PIL import Image,ImageOps

cudnn.benchmark = True

torch.manual_seed(2019)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'iSalGan'
Epoch_wise = "Epoch_wise"

args = {
    'start_epoch' : 0,
    'stop_epoch' : 30,
    'iter_num': 6000,
    'train_batch_size': 8,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': ''
}

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(300),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

train_set = ImageFolder(msra10k_path, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True)


log_path = os.path.join(ckpt_path, exp_name,Epoch_wise, str(datetime.datetime.now()) + '.txt')
#log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


criterion = nn.BCEWithLogitsLoss().cuda()



generator = iSalGan().cuda().train()

discriminator = Discriminator().cuda()

g_optimizer = optim.SGD([
        {'params': [param for name, param in generator.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in generator.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])


print('loading generator weights')
generator.load_state_dict(torch.load("/home/gautam/Project/iSalGan/ckpt/iSalGan/%generator.pth"))

print('loading generator\'s optimizatior weights')
g_optimizer.load_state_dict(torch.load("/home/gautam/Project/iSalGan/ckpt/iSalGan/%optim.pth"))


#d_optimizer = optim.Adagrad(discriminator.parameters(), lr=args['lr'])
d_optimizer = optim.SGD([
        {'params': [param for name, param in generator.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in generator.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])


if len(args['snapshot']) > 0:
    print ('training resumes from ' + args['snapshot'])
    generator.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
    g_optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
    g_optimizer.param_groups[0]['lr'] = 2 * args['lr']
    g_optimizer.param_groups[1]['lr'] = args['lr']

check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
open(log_path, 'w').write(str(args) + '\n\n')

device  = torch.device("cuda")

#########################################################################################################################################

def predict(model, img, epoch, path):
    model.eval()
    new_path = path + str(epoch) + ".png"
    #print(img.size())
    predict = model(img) # Send the combined saliency as main output
    prediction = to_pil(predict[2].data.squeeze(0).cpu())
    prediction.save(new_path)
    model.train()


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

val_path = "./MSRA10K_Imgs_GT/Imgs/979.jpg"
DIR_TO_SAVE = "./ckpt/iSalGan/Generator_output/epoch_wise/"
if not os.path.exists(DIR_TO_SAVE):
    os.makedirs(DIR_TO_SAVE)

validation_sample = Image.open(val_path).convert("RGB")
validation_sample = ImageOps.fit(validation_sample,(300,300),Image.ANTIALIAS)
validation_sample = img_transform(validation_sample).unsqueeze(0).cuda()
#print(validation_sample.size())
#assert False

###############################################################################################################################################

r_label = 1
f_label = 0

real_label = torch.full((args['train_batch_size'],),r_label,device=device)
fake_label = torch.full((args['train_batch_size'],),f_label,device=device)


alpha = 0.05
start_time = time.time()
total_iteration = len(train_loader)


for epoch in range(args["start_epoch"],args["stop_epoch"]):

    for iteration, (img,gt) in tqdm(enumerate(train_loader)):
        img = img.cuda()
        gt = gt.cuda()

        if (iteration%2 == 0):
            # Training the Discriminator

            # Train with Real Samples
            d_optimizer.zero_grad()
            inp_d = torch.cat((img,gt),1)
            out = discriminator(inp_d).squeeze()
            d_real_loss = criterion(out, real_label)
            d_real_loss.backward()
            D_x = out.mean().item()

            # Train with Fake samples
            _,_,fake_gt = generator(img)
            inp_d = torch.cat((img,fake_gt),1)
            out = discriminator(inp_d).squeeze()
            d_fake_loss = criterion(out,fake_label)
            d_fake_loss.backward()
            D_G_z1 = out.mean().item()

            d_loss = d_real_loss + d_fake_loss

            d_optimizer.step()

        else:
            # Training the Generator

            g_optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(iteration) / total_iteration
                                                                ) ** args['lr_decay']
            g_optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(iteration) / total_iteration
                                                                ) ** args['lr_decay']

            g_optimizer.zero_grad()

            _,_,fake_gt = generator(img)
            inp_d = torch.cat((img,fake_gt),1)
            out = discriminator(inp_d).squeeze()

            D_G_z2 = out.mean().item()

            g_gen_loss = criterion(fake_gt,gt)
            g_dis_loss = criterion(out.detach(),real_label)
            g_loss = torch.sum(g_dis_loss + alpha*g_gen_loss)

            g_loss.backward()

            g_optimizer.step()


            if (iteration+1)%1 == 0:
                log = "\nEpoch: {}/{}, Step: {}/{}\nd_loss: {:.4f}, g_loss: {:.4f}\nD(x): {:.4f}, D(G(x)): {:.4f}/{:.4f}\ntime: {:.4f}\n " \
                   .format( epoch, args["stop_epoch"], iteration+1, total_iteration, d_loss.item(), g_loss.item(), D_x, D_G_z1,D_G_z2, time.time()-start_time)
                print(log)
                open(log_path, 'a').write(log + '\n')

    predict(generator, validation_sample, epoch, DIR_TO_SAVE)

    torch.save(generator.state_dict(), os.path.join(ckpt_path, exp_name,Epoch_wise, 'iSalGan_generator{}.pth'.format(epoch) ))
    torch.save(discriminator.state_dict(), os.path.join(ckpt_path, exp_name,Epoch_wise, 'iSalGan_discriminator{}.pth'.format(epoch) ))
    torch.save(g_optimizer.state_dict(),os.path.join(ckpt_path, exp_name,Epoch_wise,'iSalGan_g_optim.pth{}'.format(epoch)))
    torch.save(d_optimizer.state_dict(),os.path.join(ckpt_path, exp_name,Epoch_wise, 'iSalGan_d_optim.pth{}'.format(epoch)))

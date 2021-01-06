import datetime
import os

import torch
from torch import nn
from torch import optim

from torch.utils.data import DataLoader
from torchvision import transforms

import joint_transforms
from config import msra10k_path
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir

from iSalGan_generator import iSalGan

from torch.backends import cudnn

from tqdm import tqdm

cudnn.benchmark = True

torch.manual_seed(2019)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'iSalGan'

args = {
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


log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')

criterion = nn.BCEWithLogitsLoss().cuda()



generator = iSalGan().cuda().train()

g_optimizer = optim.SGD([
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


counter = 0
total_loss_record, loss0_record, loss1_record, loss2_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

global curr_iter
curr_iter = args['last_iter']

def train(epoch,generator, g_optimizer,curr_iter):

    global total_loss_record, loss0_record, loss1_record, loss2_record
    total_loss_record, loss0_record, loss1_record, loss2_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()


    for i, data in enumerate(train_loader):
        g_optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
        g_optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

        inputs, labels = data
        batch_size = inputs.size(0)
        inputs = inputs.cuda()
        labels = labels.cuda()

        g_optimizer.zero_grad()
        outputs0, outputs1, outputs2 = generator(inputs)
        loss0 = criterion(outputs0, labels)
        loss1 = criterion(outputs1, labels)
        loss2 = criterion(outputs2, labels)

        total_loss = loss0 + loss1 + loss2
        total_loss.backward()
        g_optimizer.step()

        total_loss_record.update(total_loss.item(), batch_size)
        loss0_record.update(loss0.item(), batch_size)
        loss1_record.update(loss1.item(), batch_size)
        loss2_record.update(loss2.item(), batch_size)


        curr_iter += 1
        if curr_iter%50 == 0:
            log = '[iter %d], [total loss %.5f], [loss0 %.5f], [loss1 %.5f], [loss2 %.5f], [lr %.13f]' % \
                  (curr_iter, total_loss_record.avg, loss0_record.avg, loss1_record.avg, loss2_record.avg,
                   g_optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')

        torch.save(generator.state_dict(), os.path.join(ckpt_path, exp_name, '%generator.pth' ))
        torch.save(g_optimizer.state_dict(),os.path.join(ckpt_path, exp_name, '%optim.pth'))

for epoch in tqdm(range(0,10)):
    print("Epoch: {} Starts\n".format(epoch))
    train(epoch,generator, g_optimizer,curr_iter)
    print("Epoch: {} Ends\n".format(epoch))


torch.save(generator.state_dict(), os.path.join(ckpt_path, exp_name, '%generator.pth' ))
torch.save(g_optimizer.state_dict(),os.path.join(ckpt_path, exp_name, '%optim.pth'))

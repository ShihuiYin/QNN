'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar, adjust_optimizer, get_loss_for_H


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--epochs', default=350, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer function used')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--evaluate', type=str, help='evaluate model FILE on validation set')
parser.add_argument('--save', type=str, help='path to save model')
parser.add_argument('--arch', type=str, default='VGG', help='model architecture')
parser.add_argument('--lr_final', type=float, default=-1, help='if positive, exponential lr schedule')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
train_loss = 0
train_acc = 0
test_loss = 0
test_acc = 0
start_epoch = args.start_epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

# Model
print('==> Building model..')
model_dict = {
        'VGG_a1_w1': VGG_quant('VGG', 1, 1),
        'VGG_a2_w1': VGG_quant('VGG', 2, 1),
        'VGG_a2_w2': VGG_quant('VGG', 2, 2),
        'VGGT_a1_w1': VGG_quant('VGGT', 1, 1, 256),
        'VGGT_a2_w2': VGG_quant('VGGT', 2, 2, 256),
        'VGGT_a2_w1': VGG_quant('VGGT', 2, 1, 256),
        'VGG': VGG('VGG'),
        'VGG16': VGG('VGG16'),
        'ResNet18_a1_w1': ResNet18_quant(1, 1., 1, 1.),
        'ResNet18_a2_w1': ResNet18_quant(1, 1., 2, 1.),
        'ResNet18_a4_w1': ResNet18_quant(1, 1., 4, 1.),
        'ResNet18_a4_w4': ResNet18_quant(4, 1., 4, 1.),
        'ResNet18': ResNet18()
        }
model = model_dict[args.arch]
#model = VGG_binary('VGG')
#model = VGG('VGG16')
# model = ResNet18()
# model = PreActResNet18()
# model = GoogLeNet()
# model = DenseNet121()
# model = ResNeXt29_2x64d()
# model = MobileNet()
# model = MobileNetV2()
# model = DPN92()
# model = ShuffleNetG2()
# model = SENet18()
# model = ShuffleNetV2(1)
# model = EfficientNetB0()
if args.evaluate is None:
    print(model)

regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
                                     'lr': args.lr},
                                 150: {'lr': args.lr / 10.},
                                 250: {'lr': args.lr / 100.},
                                 350: {'lr': args.lr / 1000.}})

# update regime if exponential learning rate activated
if not hasattr(model, 'regime'):
    if args.lr_final > 0:
        decay_factor = (args.lr_final / args.lr) ** (1/ (args.epochs - 1.))
        lr = args.lr
        for e in range(1, args.epochs):
            regime[e] = {'lr': lr * decay_factor}
            lr *= decay_factor

model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

if args.evaluate:
    model_path = args.evaluate
    if not os.path.isfile(model_path):
        parser.error('invalid checkpoint: {}'.format(model_path))
    checkpoint = torch.load(model_path)
    if 'net' in checkpoint:
        state_dict_name = 'net'
    elif 'model' in checkpoint:
        state_dict_name = 'model'
    else:
        state_dict_name = 'state_dict'
    model.load_state_dict(checkpoint[state_dict_name], strict=False)
    print(model)
    args.save = None
elif args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr)

# Training
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%%'
            % (train_loss/(batch_idx+1), 100.*correct/total))
    train_acc = 100.*correct/total
    train_loss = train_loss / batch_idx
    return train_loss, train_acc

def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%%'
                % (test_loss/(batch_idx+1), 100.*correct/total))

    # Save checkpoint.
    test_acc = 100.*correct/total
    test_loss = test_loss/batch_idx
    acc = 100.*correct/total
    state = {
        'model': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if acc > best_acc and args.save:
        torch.save(state, os.path.join('./checkpoint/', args.save))
        best_acc = acc
    if args.save:
        torch.save(state, os.path.join('./checkpoint/', 'model.pth'))
    return test_loss, test_acc

if args.evaluate:
    test(0)
    exit()

for epoch in range(start_epoch, args.epochs):
    optimizer = adjust_optimizer(optimizer, epoch, regime)
    if epoch == start_epoch:
        print(optimizer)
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    print('Epoch: %d/%d | LR: %.4f | Train Loss: %.3f | Train Acc: %.2f | Test Loss: %.3f | Test Acc: %.2f (%.2f)' %
            (epoch+1, args.epochs, optimizer.param_groups[0]['lr'], train_loss, train_acc, test_loss, test_acc, best_acc))

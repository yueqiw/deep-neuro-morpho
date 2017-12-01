import argparse
import os
import shutil
import time
import datetime
import numpy as np
import pandas as pd
import json
#import scipy.sparse as sp_sparse
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
import sys
sys.path.append('../')

from src.models import *
from src.data_utils import *

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value
pd.set_option('precision', 3)

np_dtype = 'float32'
cpu_dtype = torch.FloatTensor
gpu_dtype = torch.cuda.FloatTensor

print(torch.__version__ + '\n')

use_gpu = torch.cuda.is_available()
print('GPU: ' + str(use_gpu))

parser = argparse.ArgumentParser(description='Deep Neuro Morphology')
parser.add_argument('--dataset', default='rodent_256_scale', type=str,
                    help='dataset')
parser.add_argument('--model', default='resnet18', type=str,
                    help='model')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--lr-decay', default=0.5, type=float,
                    help='learning rate decay rate')
parser.add_argument('--decay-every', default=5, type=int,
                    help='decay learning rate every several epochs')
parser.add_argument('--topn-class', default=7, type=int,
                    help='use top n popular classes for learning')
parser.add_argument('--augment', dest='augment', action='store_true',
                    help='whether to use standard augmentation (default: False)')
parser.set_defaults(augment=False)

parser.add_argument('--sparse', dest='sparse', action='store_true')
parser.add_argument('--nosparse', dest='sparse', action='store_false')
parser.set_defaults(sparse=False)

parser.add_argument('--imagenet', dest='imagenet', action='store_true')
parser.set_defaults(imagenet=False)

best_prec1 = 0


def main():
    cudnn.benchmark = True
    global args, best_prec1
    best_error = np.Inf
    args = parser.parse_args()

    today = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M")

    dir_name = args.name + '_' + args.dataset + '_' + 'topcls-' + str(args.topn_class) + \
                '_' + args.model + '_' + 'arg-' + str(args.augment) + \
                '_drop-' + str(args.droprate) + '_lr' + str(args.lr) + \
                '_decay-' + str(args.decay_every) + '-' + str(args.lr_decay) + \
                '_' + today

    args_dict = vars(args)
    args_dict['time'] = today
    args_dict['dir_name'] = dir_name

    if args.tensorboard:
        configure("../runs/%s"%(dir_name))
        with open('../runs/%s/argparse.json'%(dir_name), 'w') as f:
            json.dump(args_dict, f)

    root_dir = os.path.expanduser('~/Dropbox/lib/deep_neuro_morpho/data')
    if args.dataset == 'rodent_256_scale':
        data_dir = 'png_mip_256_fit_2d'
    else:
        raise ValueError('Unknown dataset.')

    classes = np.arange(args.topn_class)
    metadata = pd.read_pickle('../data/rodent_3d_dendrites_br-ct-filter-3_all_mainclasses_use_filter.pkl')
    metadata = metadata[metadata['label1_id'].isin(classes)]
    neuron_ids = metadata['neuron_id'].values
    labels = metadata['label1_id'].values


    if args.imagenet == True:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Lambda(lambda x: x/255.0)

    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])


    train_ids, test_ids, train_y, test_y = \
        train_test_split(neuron_ids, labels, test_size=0.15, random_state=42, stratify=labels)

    train_ids, val_ids, train_y, val_y = \
        train_test_split(train_ids, train_y, test_size=0.15, random_state=42, stratify=train_y)

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        NeuroMorpho(root_dir, data_dir, train_ids, train_y, img_size=256,
                         transform=transform_train, rgb=args.imagenet),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        NeuroMorpho(root_dir, data_dir, val_ids, val_y, img_size=256,
                         transform=transform_test, rgb=args.imagenet),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        NeuroMorpho(root_dir, data_dir, test_ids, test_y, img_size=256,
                         transform=transform_test, rgb=args.imagenet),
        batch_size=args.batch_size, shuffle=True, **kwargs)


    if args.resume == '':
        if args.model == 'vggplus1':
            model = VGGplus1(num_classes=args.topn_class)
        elif args.model == 'resnet18':
            model = models.resnet18(num_classes=args.topn_class, pretrained=False)
        elif args.model == 'resnet34':
            model = models.resnet34(num_classes=args.topn_class, pretrained=False)
        elif args.model == 'vgg13bn':
            model = models.vgg13_bn(num_classes=args.topn_class, pretrained=False)
        else:
            raise ValueError('Unknown model type.')
    else:
        model = torch.load(args.resume)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if use_gpu:
        model = model.cuda()


    criterion = nn.CrossEntropyLoss().cuda()
    # TODO: add weight for imbalanced classes
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    lr = args.lr

    for epoch in range(args.epochs):
        print('\nEpoch: {0} '.format(epoch) + '\t lr: ' + '{0:.3g}\n'.format(lr))
        t_before_epoch = time.time()
        lr = adjust_learning_rate(optimizer, lr, epoch, args.lr_decay, args.decay_every)

        train_loss, train_acc, train_acc_each = \
                train(train_loader, model, criterion, optimizer, epoch, classes)

        # evaluate on validation set
        val_loss, val_acc, val_acc_each = validate(val_loader, model, criterion, epoch, classes)

        # remember best prec@1 and save checkpoint
        is_best = val_acc > best_prec1
        best_prec1 = max(val_acc, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, dir_name)
        if args.tensorboard:
            log_value('train_loss', train_loss, epoch)
            log_value('train_acc', train_acc, epoch)
            log_value('val_loss', val_loss, epoch)
            log_value('val_acc', val_acc, epoch)
            for k in classes:
                log_value('train_acc_%d' % k, train_acc_each[k], epoch)
                log_value('val_acc_%d' % k, val_acc_each[k], epoch)
    print 'Best accuracy: ', best_prec1

def train(train_loader, model, criterion, optimizer, epoch, classes):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    batch_time = AverageMeter()

    top1_each = {i: AverageMeter() for i in  np.arange(args.topn_class)}
    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        t = time.time()
        target = target.cuda(async=True)
        input = input.cuda()
        #print input.size()
        #print target.size()
        input = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        #print input_var.size()
        # compute output
        output = model(input)

        loss = criterion(output, target_var)
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        prec1_each = [x[0] for x in accuracy_multi(output.data, target, classes, topk=(1,))]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        for k in top1_each:
            top1_each[k].update(prec1_each[k][0], torch.sum(target==k))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - t)
        t = time.time()

        if args.print_freq > 0:
            if i % args.print_freq == 0:
                print('Train: [{0}-{1:0>3}]\t'
                     'Time {batch_time.val:.2g}\t'
                      'Loss {loss.val:.4g} ({loss.avg:.4f})\t'
                      'Acc: {top1.val:.4g} ({top1.avg:.4f})\t'\
                      .format(epoch, i, \
                      batch_time=batch_time, loss=losses, \
                      top1=top1))
        del output, loss

    top1_each = [top1_each[x].avg for x in top1_each]
    print('\n * Train Error: {loss.avg:.4g}\tTrain Acc: {top1.avg:.4g}'\
            .format(loss=losses, top1=top1))
    print(pd.DataFrame({'class':classes, 'acc':top1_each}).T)
    return losses.avg, top1.avg, top1_each

def validate(val_loader, model, criterion, epoch, classes):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    batch_time = AverageMeter()

    top1_each = {i: AverageMeter() for i in  np.arange(args.topn_class)}

    model.eval()

    for i, (input, target) in enumerate(val_loader):
        t = time.time()
        target = target.cuda(async=True)
        input = input.cuda()
        input = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input)

        loss = criterion(output, target_var)
        prec1 = accuracy(output.data, target, topk=(1,), )[0]
        prec1_each = [x[0] for x in accuracy_multi(output.data, target, classes, topk=(1,))]

        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        for k in top1_each:
            top1_each[k].update(prec1_each[k][0], torch.sum(target==k))

        # measure elapsed time
        batch_time.update(time.time() - t)
        t = time.time()

        if args.print_freq > 0:
            if i % args.print_freq == 0:
                print('Val: [{0}-{1:0>3}]\t'
                     'Time {batch_time.val:.2g}\t'
                      'Loss {loss.val:.4g} ({loss.avg:.4f})\t'
                      'Acc: {top1.val:.4g} ({top1.avg:.4f})\t'\
                      .format(epoch, i, \
                      batch_time=batch_time, loss=losses, \
                      top1=top1))
        del output, loss

    top1_each = [top1_each[x].avg for x in top1_each]
    print('\n * Val Error: {loss.avg:.4g}\tVal Acc: {top1.avg:.4g}'\
            .format(loss=losses, top1=top1))
    print(pd.DataFrame({'class':classes, 'acc':top1_each}).T)
    return losses.avg, top1.avg, top1_each


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_multi(output, target, classes, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res_all = []

    for c in classes:
        res = []
        class_size = torch.sum(target==c)
        if class_size > 0:
            for k in topk:
                correct_k = correct[:k, torch.nonzero(target==c)].view(-1).float().sum(0)
                res.append(correct_k.mul_(100.0 / torch.sum(target==c)))
        else:
            res.append([None])
        res_all.append(res)
    return res_all


def save_checkpoint(state, is_best, dir_name, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "../runs/%s/"%(dir_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '../runs/%s/'%(dir_name) + 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n > 0:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, lr, epoch, lr_decay, decay_every):
    """decay learning rate every several epochs"""
    if (epoch + 1) % decay_every == 0 and epoch > 0:
        lr = lr * lr_decay
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    main()

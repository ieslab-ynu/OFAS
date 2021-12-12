""" This file is for training original model without routing modules.
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


import os
import shutil
import argparse
import time
import logging
import model_RLGate
from data import *


import torch.nn.functional as F
import itertools

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_names = sorted(name for name in model_RLGate.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(model_RLGate.__dict__[name])
                     )
print(model_names)

class BatchCrossEntropy(nn.Module):
    def __init__(self):
        super(BatchCrossEntropy, self).__init__()
    def forward(self, x, target):
        logp = F.log_softmax(x)
        target = target.view(-1,1)
        output = - logp.gather(1, target)
        #https://blog.csdn.net/weixin_36487018/article/details/112523139?ops_request_misc=%25257B%252522request%25255Fid%252522%25253A%252522161102599316780265420072%252522%25252C%252522scm%252522%25253A%25252220140713.130102334..%252522%25257D&request_id=161102599316780265420072&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-112523139.pc_search_result_no_baidu_js&utm_term=.gather
        return output.detach()

def parse_args():
    # hyper-parameters are from ResNet paper
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR10 training with gating')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('arch', metavar='ARCH',
                        default='cifar10_rnn_gate_rl_38',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: cifar10_rnn_gate_rl_38)')
    parser.add_argument('--gate-type', type=str, default='ff',
                        choices=['ff'], help='gate type')
    parser.add_argument('--dataset', '-d', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'svhn'],
                        help='dataset type')
    parser.add_argument('--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--iters', default=1200, type=int,
                        help='number of total iterations (default: 600)')
    parser.add_argument('--start-iter', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--step-ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--warm-up', action='store_true',
                        help='for n = 18, the model needs to warm up for 400 '
                             'iterations')
    parser.add_argument('--save-folder', default='save_checkpoints',
                        type=str,
                        help='folder to save the checkpoints')
    parser.add_argument('--eval-every', default=100, type=int,
                        help='evaluate model every (default: 100) iterations')
    parser.add_argument('--verbose', action="store_true",
                        help='print layer skipping ratio at training')
    # rl params
    parser.add_argument('--alpha', default=0.1, type=float,
                        help='Reward magnitude for the '
                             'average number of skipped layers')
    parser.add_argument('--rl-weight', default=0.1, type=float,
                        help='rl weight')
    parser.add_argument('--gamma', default=0.9, type=float,
                        help='discount factor, default: (0.99)')
    parser.add_argument('--gammp', default=0.99, type=float,
                        help='discount factor, default: (0.99)')
    parser.add_argument('--restart', action='store_true',
                        help='restart training')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    save_path = args.save_path = os.path.join(args.save_folder, args.arch)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # config logging file
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)

    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {}'.format(
            args.arch, args.resume))
        test_model(args)


def run_training(args):
    # create model
    model = model_RLGate.__dict__[args.arch]().to(device)
    best_prec1 = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
                args.resume, checkpoint['epoch']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = True

    train_loader = prepare_train_data(dataset=args.dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    batch_criterion = BatchCrossEntropy().to(device)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       model.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_rewards = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    skip_ratios = ListAverageMeter()
    for epoch in range(args.start_iter, args.iters):
        model.train()
        end = time.time()
        for step, data in enumerate(train_loader, start=0):
            adjust_learning_rate(args, optimizer, step)
            rl_weight = adjust_rl_rate(args, step)
            input, target = data
            # measuring data loading time
            data_time.update(time.time() - end)
            target=target.to(device)
            input_var = input.to(device)
            target_var = target

            # compute output
            output, masks = model(input_var, target_var)
            action = model.saved_assactions
            actions = model.saved_actions
            rewards = {k: [] for k in range(len(actions))}
            dists = model.saved_dists
            inputs = model.saved_outputs
            targets = model.saved_targets

            skips = [mask.data.le(0.5).float().mean() for mask in masks]
            if skip_ratios.len != len(skips):
                skip_ratios.set_len(len(skips))

            # collect prediction loss for the last action.
            pred_losses = {}
            for idx in range(len(inputs)):
                # gather output and targets for each device
                pred_losses[idx] = batch_criterion(inputs[idx], targets[idx])
            loss = criterion(output, target_var)
            normalized_skip_weight = args.alpha / (len(actions[0]))
            cum_rewards = {k: [] for k in range(len(actions))}

            for idx in range(len(actions)):
                for act in actions[idx]:
                    rewards[idx].append((act.squeeze().float())
                                        * normalized_skip_weight)
                r_pre = pred_losses[idx].squeeze() * normalized_skip_weight
                R = 0
                for r_skip in rewards[idx][::-1]:
                    r_pre = r_pre * args.gammp
                    R = -r_skip + r_pre + R * args.gamma
                    cum_rewards[idx].insert(0, R.view(-1, 1))
                    # calculate losses
            rl_losses = []
            for idx in range(len(actions)):
                dist=dists[idx].log_prob(action[idx])
                dist=dist.t()
                for idy in range(len(actions[0])):
                    _loss = (-dist[idy] * (cum_rewards[idx][idy] * rl_weight))
                    rl_losses.append(_loss.mean())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            torch.autograd.backward(rl_losses + [loss])
            optimizer.step()
            # measure elapsed time
            # measure accuracy and record loss
            prec1, = accuracy(output.data, target, topk=(1,))
            total_rewards.update(torch.cat(list(itertools.chain.from_iterable(
                cum_rewards.values()))).mean().data.item(),
                                 input.size(0))
            total_gate_rewards = torch.cat(list(itertools.chain.from_iterable(
                rewards.values()))).sum().data.item()

            batch_time.update(time.time() - end)
            top1.update(prec1.item(), input.size(0))
            skip_ratios.update(skips, input.size(0))
            losses.update(loss.item(), input.size(0))
            end = time.time()

            if step % args.print_freq == 0:
                logging.info("epoch: [{0}]\t"
                             "step: [{1}/{2}]\t"
                             "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                             "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                             "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                             "Total rewards {rewards.val:.3f} ({rewards.avg:.3f})\t"
                             "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t".format(
                    epoch,
                    step,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    rewards=total_rewards,
                    top1=top1)
                )
        logging.info('total gate rewards = {:.3f}'.format(
            total_gate_rewards))

        for idx in range(skip_ratios.len):
            logging.info(
                "{} layer skipping = {:.3f}({:.3f})".format(
                    idx,
                    skip_ratios.val[idx],
                    skip_ratios.avg[idx],
                )
            )
        skip_summaries = []
        for idx in range(skip_ratios.len):
            # logging.info(
            #     "{} layer skipping = {:.3f}".format(
            #         idx,
            #         skip_ratios.avg[idx],
            #     )
            # )
            skip_summaries.append(1-skip_ratios.avg[idx])
        # compute `computational percentage`
        cp = ((sum(skip_summaries)) / (len(skip_summaries))) * 100
        logging.info('*** Computation Percentage: {:.3f} %'.format(cp))
        skip_ratios.reset()
        losses.reset()
        top1.reset()
        total_rewards.reset()
        batch_time.reset()

        prec1 = validate(args, test_loader, model, criterion)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        checkpoint_path = os.path.join(args.save_path,
                                       'checkpoint_{:05d}.pth.tar'.format(
                                           epoch))
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        },
            is_best, filename=checkpoint_path)
        shutil.copyfile(checkpoint_path, os.path.join(args.save_path,
                                                      'checkpoint_latest'
                                                      '.pth.tar'))


def validate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    skip_ratios = ListAverageMeter()
    # switch to evaluation mode
    model.eval()
    end = time.time()
    for step, data in enumerate(test_loader, start=0):
        input, target = data

        target = target.to(device)
        input_var = input.to(device)
        target_var = target

        # compute output
        output, masks = model(input_var, target_var)
        skips = [mask.data.le(0.5).float().mean() for mask in masks]
        if skip_ratios.len != len(skips):
            skip_ratios.set_len(len(skips))
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        skip_ratios.update(skips, input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (step % args.print_freq == 0) or (step == len(test_loader) - 1):
            logging.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                    step, len(test_loader), batch_time=batch_time,
                    loss=losses, top1=top1
                )
            )
    logging.info(' * Prec@1 {top1.avg:.3f}, Loss {loss.avg:.3f}'.format(
        top1=top1, loss=losses))
    for idx in range(skip_ratios.len):
        logging.info(
            "{} layer skipping = {:.3f}({:.3f})".format(
                idx,
                skip_ratios.val[idx],
                skip_ratios.avg[idx],
            )
        )
    skip_summaries = []
    for idx in range(skip_ratios.len):
        # logging.info(
        #     "{} layer skipping = {:.3f}".format(
        #         idx,
        #         skip_ratios.avg[idx],
        #     )
        # )
        skip_summaries.append(1-skip_ratios.avg[idx])
    # compute `computational percentage`
    cp = ((sum(skip_summaries)) / (len(skip_summaries))) * 100
    logging.info('*** Computation Percentage: {:.3f} %'.format(cp))
    return top1.avg


def test_model(args):
    # create model
    model = model_RLGate.__dict__[args.arch]().to(device)

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'],strict=False)
            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
                args.resume, checkpoint['epoch']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = True

    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    criterion = nn.CrossEntropyLoss().to(device)

    validate(args, test_loader, model, criterion)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path,
                                               'model_best.pth.tar'))


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
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""
    def __init__(self):
        self.len = 10000  # set up the maximum length
        self.reset()

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count

def adjust_learning_rate(args, optimizer, _iter):
    """divide lr by 10 at 32k and 48k """
    if args.warm_up and (_iter < 100):
        lr = 0.01
    elif 50 <= _iter < 100:
        lr = args.lr * (args.step_ratio ** 1)
    elif _iter >= 100:
        lr = args.lr * (args.step_ratio ** 2)
    else:
        lr = args.lr

    if _iter % args.eval_every == 0:
        logging.info('step [{}] learning rate = {}'.format(_iter, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_rl_rate(args, _iter):
    """divide lr by 10 at 32k and 48k """
    if _iter < 25:
        rlr = args.rl_weight
    elif 25 <= _iter < 75:
        rlr = args.rl_weight * (args.step_ratio ** 1)
    elif 75 <= _iter < 125:
        rlr = args.rl_weight * (args.step_ratio ** 2)
    elif _iter >= 125:
        rlr = args.rl_weight * (args.step_ratio ** 3)
    if _iter % args.eval_every == 0:
        logging.info('step [{}] rl_weight rate = {}'.format(_iter, rlr))
    return rlr


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


if __name__ == '__main__':
    main()

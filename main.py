import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

import utilities
from model import Glow
from tqdm import tqdm
from dataset import get_dataset
import math

@torch.enable_grad()
def train(epoch, model, trainloader, device, optimizer, scheduler, loss_fn, max_grad_norm):
    global global_step
    print('\nEpoch: %d' % epoch)
    model.train()
    loss_meter = utilities.AvgMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.to(device)
            optimizer.zero_grad()
            z, sldj = model(x, reverse=False)
            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            if max_grad_norm > 0:
                utilities.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()
            scheduler.step(global_step)

            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=utilities.bits_per_dimension(x, loss_meter.avg),
                                     lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(x.size(0))
            global_step += x.size(0)


@torch.no_grad()
def test(epoch, model, testloader, device, loss_fn, args):
    global best_loss
    model.eval()
    best = False
    loss_meter = utilities.AvgMeter()
    with tqdm(total=len(testloader.dataset)) as progress_bar:
        for x, _ in testloader:
            x = x.to(device)
            z, sldj = model(x, reverse=False)
            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=utilities.bits_per_dimension(x, loss_meter.avg))
            progress_bar.update(x.size(0))

    # Save checkpoint
    if loss_meter.avg < best_loss:
        print('Updating best loss: [{}] -> [{}]'.format(best_loss, loss_meter.avg))
        best_loss = loss_meter.avg
        # indicating this epoch has achieved the best loss value so far
        best = True

    # save checkpoint file on interval
    if epoch % args.ckpt_interval == 0:
        print('Saving checkpoint file from the epoch #{}'.format(epoch))
        utilities.save_model_checkpoint(model, epoch, args.dataset, loss_meter.avg, best)

    # Save samples and data on the specified interval
    if epoch % args.img_interval == 0:
        print("Saving images from the epoch #{}".format(epoch))
        os.makedirs('grids', exist_ok=True)
        # getting a sample of n images
        images = utilities.sample(model, device, args)
        # creating a path to an epoch directory so the images are sorted by epoch
        path_to_images = 'samples/epoch_{}'.format(epoch)
        utilities.save_sampled_images(epoch, images, args.num_samples, path_to_images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glow on CIFAR-10')

    # parsing args for easier running of the program
    parser = argparse.ArgumentParser()
    
    # model parameters
    parser.add_argument('--model', type=str, default='glow', help='Name of the model in use.')
    parser.add_argument('--hidden_layers', type=int, default=512, help='Number of channels.')
    parser.add_argument('--num_levels', type=int, default=3, help='Number of flow levels.')
    parser.add_argument('--num_steps', type=int, default=3, help='Number of flow steps.')
    # optimizer and scheduler parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--max_grad_norm', type=float, default=-1., help="Maximum value of gradient.")
    parser.add_argument('--max_grad_clip', type=float, default=0, help="Maximum value of gradient.")
    parser.add_argument('--sched_warmup', type=int, default=500000, help='Warm-up period for scheduler.')
    # training parameters
    parser.add_argument('--no_gpu', action='store_true', default=False, help='Flag indicating no GPU use.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--load_model', action='store_true', default=False, help='Flag indicating loading a model from specified checkpoint.')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--usage_mode', type=str, default='train', help='What mode to run the program in [train/sample] When sampling a path to a checkpoint file MUST be specified.')
    # dataset 
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10', 'chest_xray'], help='Choose dataset: [mnist/cifar10/chest_xray]')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for datasets.')
    parser.add_argument('--download', action='store_true', default=False, help='Flag indicating when a dataset should be downloaded.')
    # checkpointing and img saving
    parser.add_argument('--ckpt_interval', type=int, default=1, help='Create a checkpoint file every N epochs.')
    parser.add_argument('--img_interval', type=int, default=1, help='Generate images every N epochs.')
    parser.add_argument('--ckpt_path', type=str, default='NONE', help='Path to the checkpoint file to use.')
    # image params 
    parser.add_argument('--num_features', type=int, default=3, help='Number of spatial channels of an image [cifar10: 3 / mnist: 1].')
    parser.add_argument('--img_height', type=int, default=32, help='Image height in pixels [cifar10: 32 / mnist: 28]')
    parser.add_argument('--img_width', type=int, default=32, help='Image width in pixels [cifar10: 32 / mnist: 28]')

    args = parser.parse_args()

    best_loss = math.inf
    global_step = 0

    # Set up main device and scale batch size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get data for training according to the specified dataset name
    trainset, trainloader, testset, testloader = get_dataset(args.dataset, args.download, args.batch_size, args.num_workers)
    
    # Model
    print('Building model..')
    model = Glow(num_channels=args.hidden_layers,
               num_levels=args.num_levels,
               num_steps=args.num_steps)
    model = model.to(device)

    start_epoch = 0
    if args.load_model:
        # Load checkpoint.
        print('Resuming from checkpoint at ckpts/best.pth.tar...')
        assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('ckpts/best.pth.tar')
        model.load_state_dict(checkpoint['model'])
        best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch']
        global_step = start_epoch * len(trainset)

    loss_fn = utilities.NLLLoss().to(device)
    # optimizer takes care of updating the parameters of the model after each batch
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler takes care of the adjustment of the learning rate
    scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / args.sched_warmup))


    for epoch in range(start_epoch, start_epoch + args.epochs):
        train(epoch, model, trainloader, device, optimizer, scheduler,
              loss_fn, args.max_grad_norm)
        test(epoch, model, testloader, device, loss_fn, args)


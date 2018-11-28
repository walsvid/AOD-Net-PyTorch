import os
import random

import torch
import torch.backends.cudnn
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torchvision import transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from PIL import Image
from utils import logger
from config import get_config
from model import AODnet
from data import HazeDataset
import dataloader


# @logger
# def set_random_seed(cfg):
#     cfg.manualSeed = random.randint(1, 10000)
#     random.seed(cfg.manualSeed)
#     torch.manual_seed(cfg.manualSeed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(cfg.manualSeed)
#     print('Random Seed: ', cfg.manualSeed)


@logger
def load_data(cfg):
    data_transform = transforms.Compose([
        transforms.Resize([480, 640]),
        transforms.ToTensor(),
        # lambda x: x[:3, ::],
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    haze_dataset = HazeDataset(cfg.ori_data_path, cfg.haze_data_path, data_transform)
    loader = torch.utils.data.DataLoader(haze_dataset, batch_size=cfg.batch_size, shuffle=True,
                                         num_workers=cfg.threads, drop_last=True, pin_memory=True)
    return loader, len(loader)


@logger
def save_model(epoch, path, net, optimizer, net_name):
    if not os.path.exists(os.path.join(path, net_name)):
        os.mkdir(os.path.join(path, net_name))
    torch.save({'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},
               f=os.path.join(path, net_name, '{}_{}.pkl'.format('AOD', epoch)))


@logger
def load_network(device):
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    net = AODnet().to(device)
    net.apply(weights_init)
    return net


@logger
def load_optimizer(net, cfg):
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return optimizer


@logger
def loss_func(device):
    criterion = torch.nn.MSELoss().to(device)
    return criterion


@logger
def load_summaries(cfg):
    summary = SummaryWriter(log_dir=cfg.log_dir, comment='')
    return summary


def main(cfg):
    # -------------------------------------------------------------------
    # basic config
    print(cfg)
    if cfg.gpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # -------------------------------------------------------------------
    # load summaries
    summary = load_summaries(cfg)
    # -------------------------------------------------------------------
    # load data
    train_loader, train_number = load_data(cfg)
    # -------------------------------------------------------------------
    # load loss
    criterion = loss_func(device)
    # -------------------------------------------------------------------
    # load network
    network = load_network(device)
    # -------------------------------------------------------------------
    # load optimizer
    optimizer = load_optimizer(network, cfg)
    # -------------------------------------------------------------------
    # start train
    print('Start train')
    network.train()
    for epoch in range(cfg.epochs):
        for step, (ori_image, haze_image) in enumerate(train_loader):
            count = epoch * train_number + (step + 1)
            ori_image, haze_image = ori_image.to(device), haze_image.to(device)
            dehaze_image = network(haze_image)
            loss = criterion(dehaze_image, ori_image)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            summary.add_scalar('loss', loss.item(), count)
            if step % cfg.print_gap == 0:
                summary.add_image('DeHaze_Images', make_grid(dehaze_image[:4].data, normalize=True, scale_each=True), count)
                summary.add_image('Haze_Images', make_grid(haze_image[:4].data, normalize=True, scale_each=True), count)
                summary.add_image('Origin_Images', make_grid(ori_image[:4].data, normalize=True, scale_each=True), count)
            print('Epoch: {}/{}  |  Step: {}/{}  |  lr: {:.6f}  | Loss: {:.6f}'
                  .format(epoch+1, cfg.epochs, step+1, train_number,
                          optimizer.param_groups[0]['lr'], loss.item()))
        save_model(epoch, cfg.model_dir, network, optimizer, cfg.net_name)
    summary.close()


if __name__ == '__main__':
    config_args, unparsed_args = get_config()
    main(config_args)

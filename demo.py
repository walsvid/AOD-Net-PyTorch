import os
import glob
import torch
import torch.backends.cudnn
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from PIL import Image
from utils import logger
from config import get_config
from model import AODnet


@logger
def make_test_data(cfg, img_path_list, device):
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([480, 640]),
        torchvision.transforms.ToTensor()
    ])
    imgs = []
    for img_path in img_path_list:
        x = data_transform(Image.open(img_path)).unsqueeze(0)
        x = x.to(device)
        imgs.append(x)
    return imgs


@logger
def load_pretrain_network(cfg, device):
    net = AODnet().to(device)
    net.load_state_dict(torch.load(os.path.join(cfg.model_dir, cfg.net_name, cfg.ckpt))['state_dict'])
    return net


def main(cfg):
    # -------------------------------------------------------------------
    # basic config
    print(cfg)
    if cfg.gpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # -------------------------------------------------------------------
    # load data
    test_file_path = glob.glob('/home/wc/workspace/AOD-Net.pytorch/test_images/*.jpg')
    test_images = make_test_data(cfg, test_file_path, device)
    # -------------------------------------------------------------------
    # load network
    network = load_pretrain_network(cfg, device)
    # -------------------------------------------------------------------
    # set network weights
    # -------------------------------------------------------------------
    # start train
    print('Start eval')
    network.eval()
    for idx, im in enumerate(test_images):
        dehaze_image = network(im)
        print(test_file_path[idx])
        torchvision.utils.save_image(torch.cat((im, dehaze_image), 0), "results/" + test_file_path[idx].split("/")[-1])


if __name__ == '__main__':
    config_args, unparsed_args = get_config()
    main(config_args)

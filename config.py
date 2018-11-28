from utils import str2bool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ori_data_path', type=str, default='ori',  help='origin image path')
parser.add_argument('--haze_data_path', type=str, default='haze',  help='haze image path')
parser.add_argument('--use_gpu', type=str2bool, default=True, help='use gpu')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader, for window set to 0')
parser.add_argument('--print_gap', type=int, default=50, help='number of batches to print average loss ')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--model_dir', type=str, default='./model')
parser.add_argument('--log_dir', type=str, default='./log')
parser.add_argument('--ckpt', type=str, default='./model/nets/net_1.pkl')
parser.add_argument('--net_name', type=str, default='nets')
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--grad_clip_norm', type=float, default=0.1)


# parser.add_argument('--pretrained_path', type=str, default=None, help='folder to model checkpoints')
# parser.add_argument('--is_train', type=str2bool, default=True)
# parser.add_argument('--start_step', type=int, default=0)
# parser.add_argument('--valDataroot', required=True, help='path to val dataset')
# parser.add_argument('--valBatchSize', type=int, default=32, help='input batch size')
# parser.add_argument('--epochSize', type=int, default=840, help='number of batches as one epoch (for validating once)')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

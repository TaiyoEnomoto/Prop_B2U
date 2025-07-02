# このファイルをメインに使いたい

from __future__ import division
import os
import logging   # loggingパッケージ：printを使わなくても自動的にログを出力してくれる
import time
import glob
import datetime
import argparse
import numpy as np
from scipy.io import loadmat, savemat

import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from arch_unet import UNet
import utils as util
from collections import OrderedDict

from util.LCNNLoss import LCNNLoss
from util.TVLoss import TVLoss

parser = argparse.ArgumentParser()
parser.add_argument("--noisetype", type=str, default="gauss25", choices=['gauss25', 'gauss5_50', 'poisson25', 'poisson30', 'poisson5_50'])
parser.add_argument('--resume', type=str)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--data_dir', type=str,
                    default='./data/train/Imagenet_val')
parser.add_argument('--val_dirs', type=str, default='./data/validation')
parser.add_argument('--save_model_path', type=str,
                    default='../experiments/results')
parser.add_argument('--log_name', type=str,
                    default='b2u_unet_gauss25_112rf20')
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=3)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--w_decay', type=float, default=1e-8)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--n_snapshot', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--patchsize', type=int, default=128)
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=2.0)
parser.add_argument("--increase_ratio", type=float, default=20.0)

opt, _ = parser.parse_known_args()  # コマンドライン引数をパースした結果を保持するオブジェクト
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
operation_seed_counter = 0
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
torch.set_num_threads(6)

# config loggers. Before it, the log will not work
opt.save_path = os.path.join(opt.save_model_path, opt.log_name, systime)
os.makedirs(opt.save_path, exist_ok=True)
util.setup_logger(
    "train",
    opt.save_path,
    "train_" + opt.log_name,
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("train")


def save_network(network, epoch, name):
    save_path = os.path.join(opt.save_path, 'models')
    os.makedirs(save_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_path = os.path.join(save_path, model_name)
    if isinstance(network, nn.DataParallel) or isinstance(
        network, nn.parallel.DistributedDataParallel
    ):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)
    logger.info('Checkpoint saved to {}'.format(save_path))


def load_network(load_path, network, strict=True):
    assert load_path is not None   # load_pathがNoneの場合AssertionErrorが発生し、プログラムが即座に停止
    logger.info("Loading model from [{:s}] ...".format(load_path))
    if isinstance(network, nn.DataParallel) or isinstance(
        network, nn.parallel.DistributedDataParallel
    ):
        network = network.module
    load_net = torch.load(load_path)  # モデルのパラメータを読み込む
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith("module."):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)
    return network

def save_state(epoch, optimizer, scheduler):
    """Saves training state during training, which will be used for resuming"""
    save_path = os.path.join(opt.save_path, 'training_states')
    os.makedirs(save_path, exist_ok=True)
    state = {"epoch": epoch, "scheduler": scheduler.state_dict(), 
                                            "optimizer": optimizer.state_dict()}
    save_filename = "{}.state".format(epoch)
    save_path = os.path.join(save_path, save_filename)
    torch.save(state, save_path)

def resume_state(load_path, optimizer, scheduler):
    """Resume the optimizers and schedulers for training"""
    resume_state = torch.load(load_path)
    epoch = resume_state["epoch"]
    resume_optimizer = resume_state["optimizer"]
    resume_scheduler = resume_state["scheduler"]
    optimizer.load_state_dict(resume_optimizer)
    scheduler.load_state_dict(resume_scheduler)
    return epoch, optimizer, scheduler

def checkpoint(net, epoch, name):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


class AugmentNoise(object):
    def __init__(self, style):
        print(style)
        if style.startswith('gauss'):  # .startswith(prefix)：文字列がprefixで指定されたフレーズで始まっていればTrue、そうでなければFalseを返す
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

    def add_train_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            # 以下の処理によりバッチごとに異なる標準偏差を持つガウスノイズを適用できるようになる
            std = std * torch.ones((shape[0], 1, 1, 1), device=x.device) # バッチサイズごとの標準偏差テンソルを作成
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0.0,
                         std=std,
                         generator=get_generator(),
                         out=noise)
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_std - min_std) + min_std
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_lam - min_lam) + min_lam
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised

    def add_valid_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


# def depth_to_space(x, block_size):
#     """
#     Input: (N, C × ∏(kernel_size), L)
#     Output: (N, C, output_size[0], output_size[1], ...)
#     """
#     n, c, h, w = x.size()
#     x = x.reshape(n, c, h * w)
#     folded_x = torch.nn.functional.fold(
#         input=x, output_size=(h*block_size, w*block_size), kernel_size=block_size, stride=block_size)
#     return folded_x


def depth_to_space(x, block_size):
    return torch.nn.functional.pixel_shuffle(x, block_size)


def generate_mask(img, width=4, mask_type='random'):
    # This function generates random masks with shape (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask = torch.zeros(size=(n * h // width * w // width * width**2, ),
                       dtype=torch.int64,
                       device=img.device)
    """
    h // width: 画像をwidth×widthのブロックに分割した時の高さ方向のブロック数
    w // width: 画像をwidth×widthのブロックに分割した時の幅方向のブロック数
    h // width * w // width: 画像全体のブロック数（何個のwidth×widthブロックができるか）
    h // width * w // width * width**2: 全ブロックの全ピクセル数（1つのブロックにwidth×width個のピクセルがある）
    n * h // width * w // width * width**2: バッチサイズを考慮した全ピクセル数

    論文の図2にあるように、width=4で4×4のブロック
    すなわち、バッチサイズを考慮した全ピクセル数を長さとした1次元テンソルを作成（0埋め）
    """

    # width=4のとき、0から16、間隔は1の数値が並ぶ1次元テンソル
    # idx_list = torch.tensor([0, 1, 2, ..., 15])
    idx_list = torch.arange(  # torch.arange：指定した範囲で等間隔の数値を持つ1次元テンソルを作成する関数
        0, width**2, 1, dtype=torch.int64, device=img.device)
    # バッチサイズを考慮した全ブロック数を長さとした1次元テンソルを作成（0埋め）
    rd_idx = torch.zeros(size=(n * h // width * w // width, ),
                         dtype=torch.int64,
                         device=img.device)

    if mask_type == 'random':
        torch.randint(low=0,
                      high=len(idx_list),  # =16
                      size=(n * h // width * w // width, ),  # バッチサイズを考慮した全ブロック数
                      device=img.device,
                      generator=get_generator(device=img.device),
                      out=rd_idx)  # outは出力テンソル、つまり乱数の生成結果をrd_idxに直接書き込む
    elif mask_type == 'batch':
        rd_idx = torch.randint(low=0,
                               high=len(idx_list),
                               size=(n, ),  # バッチサイズ
                               device=img.device,
                               # tensor.repeat(size): テンソルの要素を複製して新しいサイズのテンソルを作成、sizeは各次元での繰り返し回数を指定する整数のタプル
                               # この場合、バッチサイズnのテンソルを h // width * w // width 回繰り返す
                               generator=get_generator(device=img.device)).repeat(h // width * w // width)
    elif mask_type == 'all':
        rd_idx = torch.randint(low=0,
                               high=len(idx_list),
                               size=(1, ),
                               device=img.device,
                               # サイズ1のテンソルを n * h // width * w // width 回繰り返す
                               generator=get_generator(device=img.device)).repeat(n * h // width * w // width)
    elif 'fix' in mask_type:
        index = mask_type.split('_')[-1]  # アンダースコア (_) で分割し、最後の要素を取得
        index = torch.from_numpy(np.array(index).astype(
            np.int64)).type(torch.int64)
        rd_idx = index.repeat(n * h // width * w // width).to(img.device)

    rd_pair_idx = idx_list[rd_idx]  # rd_idx の各要素をインデックスとして idx_list の対応する値を取り出す
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // width * w // width * width**2,
                                step=width**2,  # 1ブロックごとのステップ
                                dtype=torch.int64,
                                device=img.device)

    """
    rd_pair_idxは画像の4×4ブロック内のピクセル番号を表しており、0から15の範囲にある。
    しかし、このままでは画像全体のピクセル位置には対応していない。
    そこで、torch.arange()によって、ブロックごとの開始位置（オフセット）を計算し、それをrd_pair_idxに加えることで、画像全体の座標へとマッピングする
    """

    # ランダムに選ばれたピクセルだけをマスクする2次元マスクを作成（まだ1次元）
    # ランダムに選ばれたピクセルだけを1に、他は0
    mask[rd_pair_idx] = 1

    """
    1. 1次元のマスクを4次元にreshape
    2. permuteで軸を並べ替え
    3. 型をint64に戻す
    """
    mask = depth_to_space(mask.type_as(img).view(
        n, h // width, w // width, width**2).permute(0, 3, 1, 2), block_size=width).type(torch.int64)

    return mask


def interpolate_mask(tensor, mask, mask_inv):
    n, c, h, w = tensor.shape
    device = tensor.device  # "cuda:0"のようなtorch.device型のオブジェクトが入る
    mask = mask.to(device)

    # 形状(3,3)のカーネルの作成
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])

    """
    np.newaxisを使うと次元を増やすことができる
    この場合、(3,3)のnumpy arrayが[np.newaxis, np.newaxis, :, :]によって、(1,1,3,3)となる
    """
    kernel = kernel[np.newaxis, np.newaxis, :, :]

    # kernelをtensorに変換、kernelを正規化
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    # tensorに2次元畳み込みを適用
    filtered_tensor = torch.nn.functional.conv2d(
        tensor.view(n*c, 1, h, w),
        kernel,
        stride=1,
        padding=1
    )

    """
    view_as(tensor): tensorと同じ形にリシェイプ(view)するPyTorchの関数
    """
    return filtered_tensor.view_as(tensor) * mask + tensor * mask_inv

### Global MaskerとGlobal Mask Mapperの処理？ ###
class Masker(object):
    def __init__(self, width=4, mode='interpolate', mask_type='all'):
        self.width = width
        self.mode = mode
        self.mask_type = mask_type

    def mask(self, img, mask_type=None, mode=None):
        # This function generates masked images given random masks
        if mode is None:
            mode = self.mode
        if mask_type is None:
            mask_type = self.mask_type

        n, c, h, w = img.shape
        # 画像をwidth×width(4x4)のブロックに分けて、ブロック内のピクセルをランダムにマスクする
        mask = generate_mask(img, width=self.width, mask_type=mask_type)  # ランダムに選ばれた部分だけ1のマスク
        mask_inv = torch.ones(mask.shape).to(img.device) - mask   # マスク部分だけ引いて、他が1として残る
        if mode == 'interpolate':
            masked = interpolate_mask(img, mask, mask_inv)
        else:
            raise NotImplementedError

        net_input = masked  # マスク画像をちょっといじってるっぽいけどよくわからない
        return net_input, mask

    def train(self, img):
        n, c, h, w = img.shape  # n:バッチサイズ、c:チャンネル数、h:高さ、w:横
        # 形状(n, self.width**2, c, h, w)の5次元の0埋めされたテンソルを作成し、imgテンソルと同じデバイスに配置する
        tensors = torch.zeros((n, self.width**2, c, h, w), device=img.device)
        # 形状(n, self.width**2, 1, h, w)の5次元の0埋めされたテンソルを作成
        masks = torch.zeros((n, self.width**2, 1, h, w), device=img.device)
        for i in range(self.width**2):
            x, mask = self.mask(img, mask_type='fix_{}'.format(i))
            tensors[:, i, ...] = x
            masks[:, i, ...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks   # masks: Global Mask Mapper


class DataLoader_Imagenet_val(Dataset):
    def __init__(self, data_dir, patch=256):
        super(DataLoader_Imagenet_val, self).__init__()
        self.data_dir = data_dir
        self.patch = patch
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        self.train_fns.sort()
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        # random crop
        H = im.shape[0]
        W = im.shape[1]
        if H - self.patch > 0:
            xx = np.random.randint(0, H - self.patch)
            im = im[xx:xx + self.patch, :, :]
        if W - self.patch > 0:
            yy = np.random.randint(0, W - self.patch)
            im = im[:, yy:yy + self.patch, :]
        # np.ndarray to torch.tensor
        transformer = transforms.Compose([transforms.ToTensor()])
        im = transformer(im)
        return im

    def __len__(self):
        return len(self.train_fns)


class DataLoader_SIDD_Medium_Raw(Dataset):
    def __init__(self, data_dir):
        super(DataLoader_SIDD_Medium_Raw, self).__init__()
        self.data_dir = data_dir
        # get images path
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        self.train_fns.sort()
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = loadmat(fn)["x"]
        # random crop
        H, W = im.shape
        CSize = 256
        rnd_h = np.random.randint(0, max(0, H - CSize))
        rnd_w = np.random.randint(0, max(0, W - CSize))
        im = im[rnd_h : rnd_h + CSize, rnd_w : rnd_w + CSize]
        im = im[np.newaxis, :, :]
        im = torch.from_numpy(im)
        return im

    def __len__(self):
        return len(self.train_fns)


def get_SIDD_validation(dataset_dir):
    val_data_dict = loadmat(
        os.path.join(dataset_dir, "ValidationNoisyBlocksRaw.mat"))
    val_data_noisy = val_data_dict['ValidationNoisyBlocksRaw']
    val_data_dict = loadmat(
        os.path.join(dataset_dir, 'ValidationGtBlocksRaw.mat'))
    val_data_gt = val_data_dict['ValidationGtBlocksRaw']
    num_img, num_block, _, _ = val_data_gt.shape
    return num_img, num_block, val_data_noisy, val_data_gt

# coco2017をPILで開いて、numpyに変換して計算しやすくする
def validation_coco(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))  # glob.glob(): 条件を満たすパスの文字列を要素とするリストを取得
    fns.sort()    # 取得したファイルのリストを昇順（アルファベット順）にソート
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images

# kodakをPILで開いて、numpyに変換して計算しやすくする
def validation_kodak(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))  # glob.glob(): 条件を満たすパスの文字列を要素とするリストを取得
    fns.sort()    # 取得したファイルのリストを昇順（アルファベット順）にソート
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images

# bsd300をPILで開いて、numpyに変換して計算しやすくする
def validation_bsd300(dataset_dir):
    fns = []
    fns.extend(glob.glob(os.path.join(dataset_dir, "test", "*")))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images

# set14をPILで開いて、numpyに変換して計算しやすくする
def validation_Set14(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref, data_range=255.0):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(data_range**2 / np.mean(np.square(diff)))
    return psnr

# setup loss & optimization (追加)
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # CPUかGPUどっちを使うか
lambda_TV   = 10
lambda_LCNN = 10
func_TV = TVLoss().to(device)
func_LCNN = LCNNLoss(patch_size=16).to(device)

# Training Set
TrainingDataset = DataLoader_Imagenet_val(opt.data_dir, patch=opt.patchsize)
TrainingLoader = DataLoader(dataset=TrainingDataset,
                          # num_workers=8,
                            batch_size=opt.batchsize,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)

# Validation Set
Kodak_dir = os.path.join(opt.val_dirs, "Kodak24")
BSD300_dir = os.path.join(opt.val_dirs, "BSD300")
Set14_dir = os.path.join(opt.val_dirs, "Set14")
coco2017_dir = os.path.join(opt.val_dirs)
# valid_dict = {
#     "Kodak24": validation_kodak(Kodak_dir),
#     "BSD300": validation_bsd300(BSD300_dir),
#     "Set14": validation_Set14(Set14_dir)
# }

# coco2017、BSD500のどれを使うかで分ける
valid_dict = {
    "coco2017": validation_coco(coco2017_dir)  
    #"bsd500": validation_coco(coco2017_dir)
    # "Kodak24": validation_kodak(Kodak_dir)
}

# Noise adder
noise_adder = AugmentNoise(style=opt.noisetype)

# Masker
masker = Masker(width=4, mode='interpolate', mask_type='all')

# Network
network = UNet(in_channels=opt.n_channel,
                out_channels=opt.n_channel,
                wf=opt.n_feature)
if opt.parallel:
    network = torch.nn.DataParallel(network)
network = network.cuda()

# about training scheme
num_epoch = opt.n_epoch
ratio = num_epoch / 100
optimizer = optim.Adam(network.parameters(), lr=opt.lr,
                       weight_decay=opt.w_decay)
scheduler = lr_scheduler.MultiStepLR(optimizer,
                                     milestones=[
                                         int(20 * ratio) - 1,
                                         int(40 * ratio) - 1,
                                         int(60 * ratio) - 1,
                                         int(80 * ratio) - 1
                                     ],
                                     gamma=opt.gamma)
print("Batchsize={}, number of epoch={}".format(opt.batchsize, opt.n_epoch))

# Resume and load pre-trained model
epoch_init = 1
if opt.resume is not None:
    epoch_init, optimizer, scheduler = resume_state(opt.resume, optimizer, scheduler)
if opt.checkpoint is not None:
    network = load_network(opt.checkpoint, network, strict=True)

# temp
if opt.checkpoint is not None:
    epoch_init = 60
    for i in range(1, epoch_init):
        scheduler.step()
        new_lr = scheduler.get_lr()[0]
        logger.info('----------------------------------------------------')
        logger.info("==> Resuming Training with learning rate:{}".format(new_lr))
        logger.info('----------------------------------------------------')

print('init finish')

if opt.noisetype in ['gauss25', 'poisson30']:
    Thread1 = 0.8
    Thread2 = 1.0
else:
    Thread1 = 0.4
    Thread2 = 1.0

Lambda1 = opt.Lambda1
Lambda2 = opt.Lambda2
increase_ratio = opt.increase_ratio

# エポックごとに処理
for epoch in range(epoch_init, opt.n_epoch + 1):
    cnt = 0

    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

    network.train()

    # バッチサイズ個数分の画像データを取り出して、バッチごとに処理
    # trainフェーズ
    for iteration, clean in enumerate(TrainingLoader):
        st = time.time()

        # クリーン画像とノイズ画像の用意(cleanには1バッチ分の画像が入っている)
        clean = clean / 255.0  # クリーンな画像を正規化
        clean = clean.cuda()
        noisy = noise_adder.add_train_noise(clean)  # ノイズ画像を作成

        optimizer.zero_grad()

        # masker.train()にnoisyを入れることで4枚のマスク画像が生成される？

        # masker.train(): Global Masker、net_input: 図のΩ_y、mask: Global Mask Mapper  つまり、net_inputはマスク付きノイズ画像、maskはそのままマスク
        net_input, mask = masker.train(noisy)  

        # U-Netにマスク付きノイズ画像を入れる
        noisy_output = network(net_input)

        n, c, h, w = noisy.shape

        # 予測したノイズにmask適用、reshapeしてdim=1で総和を取る？これ何をしている？
        noisy_output = (noisy_output*mask).view(n, -1, c, h, w).sum(dim=1)
        diff = noisy_output - noisy  # 図のh(f_θ(Ω_y)) # 予測されたノイズ成分 - ノイズ画像？（負になるのでは？）

        # ノイズ画像yをノイズ除去ネットワークに入れてf_θ(y)を生成
        with torch.no_grad():
            exp_output = network(noisy)
        exp_diff = exp_output - noisy  # 図のf_θ(y)

        # lossの係数を算出
        # g25, p30: 1_1-2; frange-10
        # g5-50 | p5-50 | raw; 1_1-2; range-10
        Lambda = epoch / opt.n_epoch
        if Lambda <= Thread1:
            beta = Lambda2
        elif Thread1 <= Lambda <= Thread2:
            beta = Lambda2 + (Lambda - Thread1) * \
                (increase_ratio-Lambda2) / (Thread2-Thread1)
        else:
            beta = increase_ratio
        alpha = Lambda1

        loss_TV   = func_TV(diff)      # version 1
        loss_LCNN = func_LCNN(diff)
        # loss_TV   = func_TV(exp_output)    # version 2
        # loss_LCNN = func_LCNN(exp_output)   # 勾配計算されない
        # loss_TV   = func_TV(network(noisy))    # version 3
        # loss_LCNN = func_LCNN(network(noisy))


        revisible = diff + beta * exp_diff  # h(f_θ(Ω_y)) + f_θ(y)
        loss_reg = alpha * torch.mean(diff**2)  # 正則化項の計算
        loss_rev = torch.mean(revisible**2)   # revisible lossの計算
        loss_all = loss_reg + loss_rev + lambda_TV*loss_TV + lambda_LCNN*loss_LCNN

        # 元の正則化項をなくして、LCNN LossとTV Lossを入れる → 多少は正則化項同士のコンフリクトが無くなるのでは？
        #loss_all = loss_rev + lambda_TV*loss_TV + lambda_LCNN*loss_LCNN

        loss_all.backward()
        optimizer.step()
        logger.info(
            '{:04d} {:05d} diff={:.6f}, exp_diff={:.6f}, Loss_Reg={:.6f}, Lambda={}, Loss_Rev={:.6f}, Loss_LCNN={:.6f}, Loss_TV={:.6f}, Loss_All={:.6f}, Time={:.4f}'
            .format(epoch, iteration, torch.mean(diff**2).item(), torch.mean(exp_diff**2).item(),
                    loss_reg.item(), Lambda, loss_rev.item(), loss_LCNN.item(), loss_TV.item(), loss_all.item(), time.time() - st))

    scheduler.step()

    # validationフェーズ
    if epoch % opt.n_snapshot == 0 or epoch == opt.n_epoch:   # defaultでn_snapshot==1だからこの条件は絶対にTrue
        network.eval()
        # save checkpoint
        save_network(network, epoch, "model")
        save_state(epoch, optimizer, scheduler)

        # validationのファイルパス関係
        save_model_path = os.path.join(opt.save_model_path, opt.log_name,
                                       systime)
        validation_path = os.path.join(save_model_path, "validation")
        os.makedirs(validation_path, exist_ok=True)
        np.random.seed(101)
        valid_repeat_times = {"Kodak24": 10, "BSD300": 3, "Set14": 20, "coco2017": 1, "bsd500": 1}

        # items(): 各要素のキーと値に対してforループ処理
        for valid_name, valid_images in valid_dict.items():
            # 数値格納リスト初期設定
            avg_psnr_dn = []
            avg_ssim_dn = []
            avg_psnr_exp = []
            avg_ssim_exp = []
            avg_psnr_mid = []
            avg_ssim_mid = []

            # 保存ファイル設定
            save_dir = os.path.join(validation_path, valid_name)
            os.makedirs(save_dir, exist_ok=True)

            # validation繰り返し回数(coco2017では1回)
            repeat_times = valid_repeat_times[valid_name]

            for i in range(repeat_times):
                for idx, im in enumerate(valid_images):
                    origin255 = im.copy()
                    origin255 = origin255.astype(np.uint8)
                    im = np.array(im, dtype=np.float32) / 255.0  # numpyのndarrayに変換して正規化

                    # ノイズ画像作成
                    noisy_im = noise_adder.add_valid_noise(im)
                    if epoch == opt.n_snapshot:
                        noisy255 = noisy_im.copy()
                        noisy255 = np.clip(noisy255 * 255.0 + 0.5, 0,
                                           255).astype(np.uint8)
                    # padding to square
                    H = noisy_im.shape[0]
                    W = noisy_im.shape[1]
                    val_size = (max(H, W) + 31) // 32 * 32
                    noisy_im = np.pad(
                        noisy_im,
                        [[0, val_size - H], [0, val_size - W], [0, 0]],
                        'reflect')
                    
                    # 画像をTensor型に変換
                    transformer = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    noisy_im = transformer(noisy_im)

                    # torch.unsqueeze(input, dim): inputに操作を行いたいテンソルを指定、dimに新しく挿入したい次元のインデックスを指定
                    # 挿入された次元のサイズは1になる
                    noisy_im = torch.unsqueeze(noisy_im, 0)
                    noisy_im = noisy_im.cuda()

                    with torch.no_grad():
                        n, c, h, w = noisy_im.shape

                        # masker.train(): Global Masker、net_input: 図のΩ_y、mask: Global Mask Mapper
                        net_input, mask = masker.train(noisy_im)

                        noisy_output = (network(net_input) * mask).view(n, -1, c, h, w).sum(dim=1)  # h(f_θ(Ω_y))？
                        dn_output = noisy_output.detach().clone()
                        # Release gpu memory
                        del net_input, mask, noisy_output
                        torch.cuda.empty_cache()
                        exp_output = network(noisy_im)  # f_θ(y)？
                    pred_dn = dn_output[:, :, :H, :W]
                    pred_exp = exp_output.detach().clone()[:, :, :H, :W]
                    pred_mid = (pred_dn + beta*pred_exp) / (1 + beta)  # 図の一番最後の画像

                    # Release gpu memory
                    del exp_output
                    torch.cuda.empty_cache()

                    pred_dn = pred_dn.permute(0, 2, 3, 1)
                    pred_exp = pred_exp.permute(0, 2, 3, 1)
                    pred_mid = pred_mid.permute(0, 2, 3, 1)

                    pred_dn = pred_dn.cpu().data.clamp(0, 1).numpy().squeeze(0)
                    pred_exp = pred_exp.cpu().data.clamp(0, 1).numpy().squeeze(0)
                    pred_mid = pred_mid.cpu().data.clamp(0, 1).numpy().squeeze(0)

                    pred255_dn = np.clip(pred_dn * 255.0 + 0.5, 0,
                                         255).astype(np.uint8)
                    pred255_exp = np.clip(pred_exp * 255.0 + 0.5, 0,
                                          255).astype(np.uint8)
                    pred255_mid = np.clip(pred_mid * 255.0 + 0.5, 0,
                                          255).astype(np.uint8)

                    # calculate psnr
                    psnr_dn = calculate_psnr(origin255.astype(np.float32),
                                             pred255_dn.astype(np.float32))
                    avg_psnr_dn.append(psnr_dn)
                    ssim_dn = calculate_ssim(origin255.astype(np.float32),
                                             pred255_dn.astype(np.float32))
                    avg_ssim_dn.append(ssim_dn)

                    psnr_exp = calculate_psnr(origin255.astype(np.float32),
                                              pred255_exp.astype(np.float32))
                    avg_psnr_exp.append(psnr_exp)
                    ssim_exp = calculate_ssim(origin255.astype(np.float32),
                                              pred255_exp.astype(np.float32))
                    avg_ssim_exp.append(ssim_exp)

                    psnr_mid = calculate_psnr(origin255.astype(np.float32),
                                              pred255_mid.astype(np.float32))
                    avg_psnr_mid.append(psnr_mid)
                    ssim_mid = calculate_ssim(origin255.astype(np.float32),
                                              pred255_mid.astype(np.float32))
                    avg_ssim_mid.append(ssim_mid)

                    # visualization
                    if i == 0 and epoch == opt.n_snapshot:
                        save_path = os.path.join(
                            save_dir,
                            "{}_{:03d}-{:03d}_clean.png".format(
                                valid_name, idx, epoch))
                        Image.fromarray(origin255).convert('RGB').save(
                            save_path)
                        save_path = os.path.join(
                            save_dir,
                            "{}_{:03d}-{:03d}_noisy.png".format(
                                valid_name, idx, epoch))
                        Image.fromarray(noisy255).convert('RGB').save(
                            save_path)
                    if i == 0:
                        save_path = os.path.join(
                            save_dir,
                            "{}_{:03d}-{:03d}_dn.png".format(
                                valid_name, idx, epoch))
                        Image.fromarray(pred255_dn).convert(
                            'RGB').save(save_path)
                        save_path = os.path.join(
                            save_dir,
                            "{}_{:03d}-{:03d}_exp.png".format(
                                valid_name, idx, epoch))
                        Image.fromarray(pred255_exp).convert(
                            'RGB').save(save_path)
                        save_path = os.path.join(
                            save_dir,
                            "{}_{:03d}-{:03d}_mid.png".format(
                                valid_name, idx, epoch))
                        Image.fromarray(pred255_mid).convert(
                            'RGB').save(save_path)

            avg_psnr_dn = np.array(avg_psnr_dn)
            avg_psnr_dn = np.mean(avg_psnr_dn)
            avg_ssim_dn = np.mean(avg_ssim_dn)

            avg_psnr_exp = np.array(avg_psnr_exp)
            avg_psnr_exp = np.mean(avg_psnr_exp)
            avg_ssim_exp = np.mean(avg_ssim_exp)

            avg_psnr_mid = np.array(avg_psnr_mid)
            avg_psnr_mid = np.mean(avg_psnr_mid)
            avg_ssim_mid = np.mean(avg_ssim_mid)

            log_path = os.path.join(validation_path,
                                    "A_log_{}.csv".format(valid_name))
            with open(log_path, "a") as f:
                f.writelines("epoch:{},dn:{:.6f}/{:.6f},exp:{:.6f}/{:.6f},mid:{:.6f}/{:.6f}\n".format(
                    epoch, avg_psnr_dn, avg_ssim_dn, avg_psnr_exp, avg_ssim_exp, avg_psnr_mid, avg_ssim_mid))

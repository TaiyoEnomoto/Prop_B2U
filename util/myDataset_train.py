#!C:\ProgramData\Anaconda3
# coding: utf-8
import os
import random
import numpy as np
from pathlib import Path  # Path クラスは、ファイルパスやディレクトリパスをオブジェクトとして扱うことで、パスの結合やファイルの存在確認などを行える
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class myDataset(Dataset):
    IMG_EXTENSIONS = [".png", ".jpg"]

    def __init__(
            self, img_dir,
            noise_ver="gaussian",
            len_crop=256, rand_ratio=0.5,
            sigma_min=5, sigma_max=50, lambda_min=5, lambda_max=50,
            flag_aug=True, ratio=0.9, size_data=(256, 256, 3), size_window=(5, 5) # 追加
        ):
        super(myDataset, self).__init__()
        # self.img_dir   = img_dir
        self.img_paths = self.get_img_paths(img_dir)
        self.totensor = transforms.ToTensor()    # 画像をPyTorchのテンソルに変換する。この変換は、画像を0~255のピクセル値から0~1の範囲の浮動小数点数に正規化する。
        self.augumentation = transforms.Compose([  # Composeは、複数の変換を連続して適用するためのクラス。この場合、画像に対して順番にランダムクロップ、水平フリップ、垂直フリップ、回転が適用される。
            transforms.RandomCrop(size=len_crop),   # 画像をランダムな位置でクロップ（切り取り）。
            transforms.RandomHorizontalFlip(p=rand_ratio),  # 画像を確率 p=rand_ratio で左右反転
            transforms.RandomVerticalFlip(p=rand_ratio),   # 画像を確率 p=rand_ratio で上下反転
            transforms.Lambda(RandomRotation90(p=rand_ratio)),
        ])
        if noise_ver=="gaussian":
            self.noise = NoiseGaussian(sigma_min=sigma_min, sigma_max=sigma_max)
        elif noise_ver=="poisson":
            self.noise = NoisePoisson(lambda_min=lambda_min, lambda_max=lambda_max)
        self.flag_aug = flag_aug
        self.ratio = ratio
        self.size_data = size_data
        self.size_window = size_window
        
    ### end __init__

    def __getitem__(self, index):
        path = self.img_paths[index]
        # xxx = os.path.splitext(os.path.basename(path))
        # img_name = xxx[0]
        img_name = os.path.basename(path)  # パスからファイル名を取得するosモジュールのメソッド

        gt = Image.open(path)
        gt = self.totensor(gt)
        if self.flag_aug:
            gt = self.augumentation(gt)
        img = self.noise(gt)
        

        # Noise2Void のマスク生成 # 追加
        # detachを使って元のテンソルとの関係を切って新しいテンソルを作る
        img_mask, mask = self.generate_mask(img.detach()) # マスクとノイズ除去

        return img, img_mask, mask  # X(GTにノイズを入れた画像), X'(Xにマスクを適用した画像), M(マスク)の順番になっている
    ### end __getitem__

    def __len__(self):
        return len(self.img_paths)

    def get_img_paths(self, img_dir):
        img_dir = Path(img_dir)
        img_paths = [
            p for p in img_dir.iterdir() if p.suffix in myDataset.IMG_EXTENSIONS   # p.suffixを使って拡張子がpngかjpgの画像を取り出す
        ]
        return img_paths
    ### end get_img_paths

    # inputはtorch.tensor
    def generate_mask(self, input):  # train.pyで(128, 128)でcropするように指定されているから、128×128で計算できるようにする必要がある
        ratio = self.ratio
        size_window = self.size_window
        # size_data = self.size_data

        # (128, 128, 3)に変更
        size_data = (128, 128, 3)

        num_sample = int(size_data[0] * size_data[1] * (1 - ratio))

        """inputをndarrayにする

        torch.tensorはチャンネル、縦、横
        ndarrayは縦、横、チャンネル
        数値のタイプをcheack(float32からfloat32になっているかどうかprintを使って)
        """
        
        input = input.numpy()              # 以下の計算はnumpyであるためtorch.tensor型からnumpy.ndarray型に変換
        input = input.transpose(1, 2, 0)   # inputの形状変換 (チャンネル,縦,横)から(縦,横,チャンネル)に変換

        mask = np.ones(input.shape)

        mask = mask.astype(np.float32)     # maskのデータ型をfloat64からfloat32に変換

        output = input

        for ich in range(size_data[2]):
            idy_msk = np.random.randint(0, size_data[0], num_sample)
            idx_msk = np.random.randint(0, size_data[1], num_sample)

            idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2, size_window[0] // 2 + size_window[0] % 2, num_sample)
            idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2, size_window[1] // 2 + size_window[1] % 2, num_sample)

            idy_msk_neigh = idy_msk + idy_neigh
            idx_msk_neigh = idx_msk + idx_neigh

            idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[0] - (idy_msk_neigh >= size_data[0]) * size_data[0]
            idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[1] - (idx_msk_neigh >= size_data[1]) * size_data[1]

            id_msk = (idy_msk, idx_msk, ich)
            id_msk_neigh = (idy_msk_neigh, idx_msk_neigh, ich)

            output[id_msk] = input[id_msk_neigh]
            mask[id_msk] = 0.0
        
        # outputとmaskをtorch.tensorにする
        transform = transforms.ToTensor()
        mask = transform(mask)
        output = transform(output)

        return output, mask   # 近傍のピクセルをコピーして、そのピクセルで中心画素を隠したoutput、隠された画素部分を0.0にしたmaskをreturn
    ### end generate_mask
### end class

# torch.Tensorをランダムに90°回転させる
class RandomRotation90:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input):
        if random.random() < self.p:
            input = torch.rot90(input, 1, [1, 2])

        return input
### end class

# ノイズ ガウシアン
class NoiseGaussian:
    def __init__(self, sigma_min=5, sigma_max=50):
        self.sigma_min = sigma_min/255
        self.sigma_max = sigma_max/255

    def __call__(self, img):
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * random.random()
        img = img + sigma*torch.randn_like(img)
        img = torch.clamp(img, min=0, max=1)
        return img
### end class

# ノイズ ポアソン
class NoisePoisson:
    def __init__(self, lambda_min=5, lambda_max=50):
        self.lambda_min = lambda_min/255
        self.lambda_max = lambda_max/255

    def __call__(self, img):
        lambda_ = self.lambda_min + (self.lambda_max - self.lambda_min) * random.random()
        img = img + torch.poisson(torch.ones(img.size())*lambda_)
        img = torch.clamp(img, min=0, max=1)
        return img
### end class
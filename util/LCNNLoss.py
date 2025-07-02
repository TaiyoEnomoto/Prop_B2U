#
#
#
import torch
import torch.linalg as LA

from einops.layers.torch import Rearrange

# Local color nuclear norm loss function
class LCNNLoss(torch.nn.Module):
    def __init__(self, patch_size=16):
        super(LCNNLoss, self).__init__()

        """
        元々のテンソルの形は(b, c, h, w)
        そのため、bs ch (h p1) (w p2)と書くと、高さをパッチサイズごとに分割、幅をパッチサイズごとに分割することになる

        ＜元の形＞bs:バッチサイズ、ch:チャンネル数、(h p1):高さをpatch_sizeずつに分割、(w,p2):幅をpatch_sizeずつに分割
        ＜新しい形＞bs:バッチサイズ(そのまま)、ch:チャンネル数(そのまま)、(h w):パッチの総数(高さ×幅個)、(p1 p2):各パッチの中身(パッチサイズ×パッチサイズ)
        """
        self.to_patches = Rearrange(
            'bs ch (h p1) (w p2) -> bs ch (h w) (p1 p2)',
            p1=patch_size, 
            p2=patch_size
        )
        """
            入力画像の高さhと幅wが、それぞれpatch_size（p1とp2）で割り切れる必要がある
            割り切れない場合、einopsはエラーを発生させる
        """
        self.to_vectors = Rearrange(
            'bs vec ch pp -> (bs vec) ch pp'
        )


    def forward(self, img):
        (bs, ch, h, w) = img.size()
        patch = self.to_patches(img)          # bs ch (h w) (p1 p2)になる
        patch = patch.permute(0, 2, 1, 3)     # bs (h w) ch (p1 p2)になる
        patch = self.to_vectors(patch)        # (bs h w) ch (p1 p2)になる？

        # SVDの特異値を計算
        loss = LA.svdvals(patch)

        # 最大特異値（0番目の列）を0にする（in-placeではなくout-of-place操作）
        loss = torch.cat([torch.zeros(loss.shape[0], 1, device=loss.device), loss[:, 1:]], dim=1)

        loss = torch.sum(loss, 1) / (bs*ch*h*w) 
        loss = torch.sum(loss, 0)
        return loss
### end class
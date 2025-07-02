import torch

# Main class
# Total Variation loss function
class TVLoss(torch.nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, img):
        (bs, ch, h, w) = img.size()
        tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
        output = (tv_h + tv_w) / (bs*ch*h*w)
        #print(f"[DEBUG TV] output = {output}") #デバッグ（オーダー確認）
        return output
### end class

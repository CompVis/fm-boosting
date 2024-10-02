import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


def un_normalize_ims(ims):
    """ Convert from [-1, 1] to [0, 255] """
    ims = ((ims * 127.5) + 127.5).clip(0, 255).to(torch.uint8)
    return ims


def calculate_PSNR(img1, img2):
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    mse = torch.mean((img1 - img2) ** 2, dim=[1,2,3])
    psnrs = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnrs.mean()


class ImageMetricTracker(nn.Module):
    def __init__(self, num_crops: int = 4, crop_size: int = 512):
        super().__init__()
        self.ssim = SSIM(data_range=1.)
        self.ssims = []
        
        self.psnrs = []

        self.mses = []

        self.fid = FrechetInceptionDistance(
            feature=2048,
            reset_real_features=True,
            normalize=False,
            sync_on_compute=True
        )

        # whether use the fid on crops during training
        self.patch_fid = num_crops > 0
        if self.patch_fid:
            print("[ImageMetricTracker] Evaluating using patch-wise FID")
        self.num_crops = num_crops
        self.crop_size = crop_size

    def __call__(self, target, pred):
        """ Assumes target and pred in [-1, 1] range """
        real_ims = un_normalize_ims(target)
        fake_ims = un_normalize_ims(pred)

        # FID
        if self.patch_fid:
            croped_real = []
            croped_fake = []
            anchors = []
            for i in range(real_ims.shape[0]*self.num_crops):
                anchors.append(transforms.RandomCrop.get_params(
                        real_ims[0], output_size=(self.crop_size, self.crop_size)))
                
            for idx, (img_real, img_fake) in enumerate(zip(real_ims, fake_ims)):
                for i in range(self.num_crops):
                    anchor = anchors[idx*self.num_crops + i]

                    croped_real.append(TF.crop(img_real, *anchor))
                    croped_fake.append(TF.crop(img_fake, *anchor))
            
            real_ims = torch.stack(croped_real)
            fake_ims = torch.stack(croped_fake)
        
        self.fid.update(real_ims, real=True)
        self.fid.update(fake_ims, real=False)

        # SSIM, PSNR, and MSE
        self.ssims.append(self.ssim(pred/2+0.5, target/2+0.5))
        self.psnrs.append(calculate_PSNR(pred/2+0.5, target/2+0.5))
        self.mses.append(nn.functional.mse_loss(pred, target))

    def reset(self):
        self.ssims = []
        self.psnrs = []
        self.mses = []
        self.fid.reset()

    def aggregate(self):
        fid = self.fid.compute()
        ssim = torch.stack(self.ssims).mean()
        psnr = torch.stack(self.psnrs).mean()
        mse = torch.stack(self.mses).mean()
        out = dict(fid=fid, ssim=ssim, psnr=psnr, mse=mse)
        return out

import os
import glob
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
import cv2
import torch.nn as nn
from flow3d.models.pwcnet import PWCNet, get_backwarp
import models


class OpticalFlow(nn.Module):
    def __init__ (self):
        super(OpticalFlow, self).__init__()
        self.alignnet = PWCNet(load_pretrained=True, 
                            weights_path="./pretrained_dirs/pwcnet-network-default.pth")
        self.alignnet.eval()
    def forward(self, pred, target):
        with torch.no_grad():
            offset = self.alignnet(pred, target)  
        align_pred, flow_mask = get_backwarp(pred, offset)
        return align_pred*flow_mask, target*flow_mask



def psnr(img1, img2, mask=None):
    if mask is None:
        mse = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    else:
        mask_bin = (mask == 1.)
        mse = (((img1 - img2)[mask_bin]) ** 2).mean()
    metric = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return metric.item()

def scene_metrics(result_dir, gt_dir, epoch=199):
    log_dir = os.path.join(result_dir,  "metrics_pwcnet.txt")
    log_f = open(log_dir, 'a')

    lpips = models.PerceptualLoss(model='net-lin',net='alex', use_gpu=True,version=0.1)

    opticalflow = OpticalFlow().cuda()

    result_dir = os.path.join(result_dir, 'x1')
    gt_dir = os.path.join(gt_dir, 'x1')
    clips = os.listdir(result_dir)
    if "0023" in clips:
        clips.remove("0023")
    clips = sorted(clips, key=lambda x: int(x.split('_')[-1]))

    files = [['00000_img.png','00001_img.png','00002_img.png','00003_img.png','00004_img.png',
              '00005_img.png','00006_img.png','00007_img.png','00008_img.png','00009_img.png'],
             ['00010_img.png','00011_img.png','00012_img.png','00013_img.png','00014_img.png',
              '00015_img.png','00016_img.png','00017_img.png','00018_img.png','00019_img.png'],
             ['00020_img.png','00021_img.png','00022_img.png','00023_img.png','00024_img.png',
              '00025_img.png','00026_img.png','00027_img.png','00028_img.png','00029_img.png'],
             ['00030_img.png','00031_img.png','00032_img.png','00033_img.png','00034_img.png',
              '00035_img.png','00036_img.png','00037_img.png','00038_img.png','00039_img.png'],
             ['00040_img.png','00041_img.png','00042_img.png','00043_img.png','00044_img.png',
              '00045_img.png','00046_img.png','00047_img.png'],
             ]
    assert len(clips) == len(files)

    results = []
    for ii in range(0, len(clips)):
        for jj in range(0, len(files[ii])):
            result = os.path.join(result_dir, clips[ii], 'results', 'rgb_deblur_mid', '%05d'%epoch, files[ii][jj])
            results.append(result)

    gts = glob.glob(os.path.join(gt_dir, 'images_test',   '[0-9][0-9][0-9][0-9][0-9].png'))
    gts = sorted(gts, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    print(len(gts), len(results))
    assert len(gts) == len(results)

    results = results[1::2]
    gts = gts[1::2]

    full_psnr_list = []
    full_ssim_list = []
    full_lpips_list = []
    for ii in range(0, len(results)):
        result = cv2.cvtColor(cv2.imread(results[ii]), cv2.COLOR_BGR2RGB)/255.
        result = torch.FloatTensor(result).cuda().clamp(0., 1.)
        gt = cv2.cvtColor(cv2.imread(gts[ii]), cv2.COLOR_BGR2RGB)/255.
        gt = torch.FloatTensor(gt).cuda().clamp(0., 1.)

        result, gt = opticalflow(result.unsqueeze(0).permute(0, 3, 1, 2), gt.unsqueeze(0).permute(0, 3, 1, 2))
        result = result[0].permute(1, 2, 0)
        gt = gt[0].permute(1, 2, 0)
        psnr_metric = psnr(gt.cpu().unsqueeze(0), result.cpu().unsqueeze(0))
        ssim_metric = structural_similarity(gt.cpu().numpy(), result.cpu().numpy(), 
                                        multichannel=True)
        lpips_metric = lpips(gt.unsqueeze(0).permute(0, 3, 1, 2)*2.-1., 
                            result.unsqueeze(0).permute(0, 3, 1, 2)*2. -1.).mean().double().item()

        full_psnr_list.append(psnr_metric)
        full_ssim_list.append(ssim_metric)
        full_lpips_list.append(lpips_metric)
    
    log_f.write('Average Result: PSNR:\t' + str(np.mean(np.array(full_psnr_list))) + '\t'+ str(np.mean(np.array(full_ssim_list))) + '\t'+str(np.mean(np.array(full_lpips_list)))+'\n')

    log_f.flush()
    log_f.close()
    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")

    parser.add_argument("--result_dir", default="", type=str)
    parser.add_argument("--gt_dir", default="", type=str)

    args = parser.parse_args()
    scene_metrics(args.result_dir, args.gt_dir) 
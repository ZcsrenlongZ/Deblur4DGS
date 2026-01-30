import os
import glob
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
import cv2
import torch.nn as nn
import models


def psnr(img1, img2, mask=None):
    if mask is None:
        mse = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    else:
        mask_bin = (mask == 1.)
        mse = (((img1 - img2)[mask_bin]) ** 2).mean()
    metric = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return metric.item()

def scene_metrics(result_dir, gt_dir, epoch=199):
    log_dir = os.path.join(result_dir,  "metrics_pose_optimization.txt")
    log_f = open(log_dir, 'a')

    lpips = models.PerceptualLoss(model='net-lin',net='alex', use_gpu=True,version=0.1)


    result_dir = os.path.join(result_dir, 'x1')
    gt_dir = os.path.join(gt_dir, 'x1')
    clips = os.listdir(result_dir)
    if "0023" in clips:
        clips.remove("0023")
    clips = sorted(clips, key=lambda x: int(x.split('_')[-1]))

    files = [['00000.png','00001.png','00002.png','00003.png','00004.png',
              '00005.png','00006.png','00007.png','00008.png','00009.png'],
             ['00010.png','00011.png','00012.png','00013.png','00014.png',
              '00015.png','00016.png','00017.png','00018.png','00019.png'],
             ['00020.png','00021.png','00022.png','00023.png','00024.png',
              '00025.png','00026.png','00027.png','00028.png','00029.png'],
             ['00030.png','00031.png','00032.png','00033.png','00034.png',
              '00035.png','00036.png','00037.png','00038.png','00039.png'],
             ['00040.png','00041.png','00042.png','00043.png','00044.png',
              '00045.png','00046.png','00047.png'],
             ]
    assert len(clips) == len(files)

    results = []
    for ii in range(0, len(clips)):
        for jj in range(0, len(files[ii])):
            result = os.path.join(result_dir, clips[ii], 'results', 'rgb_test_optim',  files[ii][jj])
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

        psnr_metric = psnr(gt.cpu().unsqueeze(0), result.cpu().unsqueeze(0))
        # ssim_metric = structural_similarity(gt.cpu().numpy(), result.cpu().numpy(), channel_axis=-1, data_range=1.0)
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
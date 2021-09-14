
import os
import matplotlib.pyplot as plt
import numpy as np
import json
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as sme
import numpy as np
import argparse
import glob
from PIL import Image


import time
from multiprocessing import Pool, Manager
from multiprocessing.pool import ThreadPool
import torch
import cv2
# from piq import ssim, SSIMLoss
# from piq import psnr

# CPU版本
def PSNR(ximg,yimg):
    return compare_psnr(ximg,yimg,data_range=255)

def SSIM(y,t,value_range=255):   
    try:
        ssim_value = ssim(y, t, gaussian_weights=True, data_range=value_range, multichannel=True)
    except ValueError:
        #WinSize too small
        ssim_value = ssim(y, t, gaussian_weights=True, data_range=value_range, multichannel=True, win_size=3)
    return ssim_value
def MSE(ximg,yimg):
    return sme(ximg,yimg)

def Evaluate(files_gt, files_pred, methods = [PSNR,MSE,SSIM]):
    score = {}
    for meth in methods:
        name = meth.__name__
        results = []
        res=0
        frame_num=len(files_gt)
        for frame in range(0,frame_num):
            # ignore some tiny crops 
            if files_gt[frame].shape[0]*files_gt[frame].shape[1]<150:    
                continue

            res = meth(files_pred[frame],files_gt[frame])
            results.append(res)        

        mres = np.mean(results)
        stdres=np.std(results)
        # print(name+": "+str(mres)+" Std: "+str(stdres))
        score['mean']=mres
        score['std']=stdres
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--groundtruth_folder',default='/home/scsc/PyTorch_Project/Github/paper_code/PyTorch/data/OurMeth/underwater1/groundtruth/real')
    parser.add_argument('--prediction_folder',default='/home/scsc/paper/ALL_Method_result/Ours/underwater1/prediction/synthesis-repair')
    #  parser.add_argument('--prediction_folder',default='/home/scsc/PyTorch_Project/Github/paper_code/PyTorch/data/OurMeth/underwater1/prediction/synthesis')
    # usage: python scoring.py --groundtruth_folder ./val --prediction_folder ./result   

    args = parser.parse_args()
    gt_root = args.groundtruth_folder
    pred_root = args.prediction_folder
    video_gt = []
    video_predict = []
    for file in sorted(os.listdir(gt_root)):
        image_gt = os.path.join(gt_root,file)
        video_gt.append(np.array(Image.open(image_gt)).astype(np.uint8))
    for file_b in sorted((os.listdir(pred_root))):
        image_predict = os.path.join(pred_root, file_b)

        video_predict.append(np.array(Image.open(image_predict)).astype(np.uint8)[:,:,0:3])
        # video_predict.append(np.array(Image.open(image_predict)).astype(np.uint8)[:,:,0:3])
    psnr_res = Evaluate(video_gt, video_predict, methods=[PSNR])
    ssim_res = Evaluate(video_gt, video_predict, methods=[SSIM])
    mse_res = Evaluate(video_gt, video_predict, methods=[MSE])

    ssim_res_norm = ssim_res['mean'] * 100



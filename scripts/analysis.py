import glob
import os
import numpy as np
import pathlib
import datetime
import time
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="../logs/*/*")
parser.add_argument("--out_path", type=str, default="out.txt")
args = parser.parse_args()

exp_name = args.exp_name

exp_list = sorted(glob.glob(exp_name))

data = {}

for exp_path in exp_list:
    check_paths = [path for path in glob.glob(exp_path + "/checkpoint*") if not "tmp" in path]
    indices = np.argsort(np.array([path.split('_')[-1] for path in check_paths], np.int64))
    check_paths = np.array(check_paths)[indices]
    check_paths = [pathlib.Path(path) for path in check_paths]
    times = []
    dtimes = []
    for i in range(9):
        dt1 = datetime.datetime.fromtimestamp(check_paths[i].stat().st_mtime)
        dt2 = datetime.datetime.fromtimestamp(check_paths[i+1].stat().st_mtime)
        dtimes.append((dt2 - dt1).seconds / 60.)
    dt = np.median(dtimes)
    times.extend(np.arange(1,11) * dt)  # 1,2,3,4,5,6,7,8,9,10
    
    dtimes = []
    for i in range(9,18):
        dt1 = datetime.datetime.fromtimestamp(check_paths[i].stat().st_mtime)
        dt2 = datetime.datetime.fromtimestamp(check_paths[i+1].stat().st_mtime)
        dtimes.append((dt2 - dt1).seconds / 10 / 60.)
    dt = np.median(dtimes)
    times.extend(np.arange(20,110,10) * dt)  # 10,20,30,40,50,60,70,80,90,100

    times = list(np.array(times) / 60.)
    print(exp_path)
    print(str(list(np.round(times, decimals=1)))[1:-1])

    
    psnrs_paths = glob.glob(exp_path + "/test_preds/psnrs_*")
    indices = np.argsort(np.array([path.split('_')[-1].split('.')[0] for path in psnrs_paths], np.int64))
    psnrs_paths = np.array(psnrs_paths)[indices]
    psnrs = []

    for path in psnrs_paths:
        with open(path, 'r') as f:
            psnr = f.readlines()
        assert len(psnr) == 1
        psnr = np.mean(np.array(psnr[0].split(), np.float32))
        psnrs.append(psnr)
    print(str(list(np.round(psnrs,3)))[1:-1], len(psnrs))

    ssims_paths = glob.glob(exp_path + "/test_preds/ssims_*")
    indices = np.argsort(np.array([path.split('_')[-1].split('.')[0] for path in ssims_paths], np.int64))
    ssims_paths = np.array(ssims_paths)[indices]
    ssims = []

    for path in ssims_paths:
        with open(path, 'r') as f:
            ssim = f.readlines()
        assert len(ssim) == 1
        ssim = np.mean(np.array(ssim[0].split(), np.float32))
        ssims.append(ssim)
    print(str(list(np.round(ssims,3)))[1:-1], len(ssims))

    data[exp_path] = (times, psnrs, ssims)

psnrs, ssims = [], []
for key, value in data.items():
    psnrs.append(value[1][-1])
    ssims.append(value[2][-1])
print("PSNR:", np.mean(psnrs), ", SSIM:", np.mean(ssims))

with open(args.out_path, 'w') as f:
    for key, value in data.items():
        f.write(key + '\n')
        for i in range(3):
            f.write(str(list(np.round(value[i], 4)))[1:-1].replace(',', '') + '\n')
        f.write('\n')

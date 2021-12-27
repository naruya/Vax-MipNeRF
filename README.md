# Vax-MipNeRF

This repository is part of the official implementation of [VaxNeRF](https://github.com/naruya/VaxNeRF).

Vax'ed MipNeRF achieves the final accuracy of the original NeRF about **28 times** faster just by combining visual hull.

![psnr_mip](https://user-images.githubusercontent.com/23403885/147459802-8871f92f-923d-437c-a0a8-79b6077985ab.png)

## Installation

Please see the README of [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf).

and please install [Gin Config](https://github.com/google/gin-config)

```
$ pip install gin-config
```

## Quick start

### Training

First, please clone the [VaxNeRF](https://github.com/naruya/VaxNeRF) repository for visual hull.
The visual hull options are a little different from VaxNeRF, so please run it again even if you have already done that before.

```shell
# make a bounding volume voxel using Visual Hull
cd /path/to/VaxNeRF
python visualhull.py \
    --config configs/demo \
    --data_dir ../data/nerf_synthetic/lego \
    --voxel_dir ../data/voxel/lego \
    --pooling 7 \
    --alpha_bkgd

# train Vax-MipNeRF
cd /path/to/Vax-MipNeRF
python train.py \
    --gin_file ./configs/blender_vax_c128f128.gin \
    --data_dir ../data/nerf_synthetic/lego \
    --voxel_dir ../data/voxel/lego \
    --train_dir ../logs/vax-mip/lego
```

### Evaluation

```shell
python eval.py \
    --gin_file ./configs/blender_vax_c128f128.gin \
    --data_dir ../data/nerf_synthetic/lego \
    --voxel_dir ../data/voxel/lego \
    --train_dir ../logs/vax-mip/lego \
    --eval_once
```

### Options

- To avoid out of memory errors, please reduce the batch size by adding the following options. (Make sure to set the `chunk` size smaller than the `batch_size`)

```shell
--gin_param="Config.batch_size=1024" --gin_param="Config.chunk=1000"
```

- To evaluate all saved checkpoints, remove the `--eval_once`. (Do not specify `--eval_once False`. It will be recognized as `True`).


## Acknowledgements
I would like to thank the authors of [NeRF](http://www.matthewtancik.com/nerf) and [MipNeRF](https://jonbarron.info/mipnerf/), and the developers of [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf).

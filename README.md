<p align="center">
 <h2 align="center">🚀 Boosting Latent Diffusion with Flow Matching</h2>
 <p align="center"> 
 Johannes S. Fischer<sup>*</sup> · Ming Gui<sup>*</sup> · Pingchuan Ma<sup>*</sup> · 
 <!-- </p>
  <p align="center">  -->
 Nick Stracke · Stefan A. Baumann ·Vincent Tao Hu · Björn Ommer
 </p>
 <p align="center"> 
    <b>CompVis Group @ LMU Munich</b>
 </p>
 </p>
  <p align="center"> <sup>*</sup> <i>equal contribution</i> </p>
</p>

<p align="center">
 <b>ECCV 2024 Oral</b>
</p>

<p align="center">
&rArr; <b>code coming soon!</b>
</p>

[![Website](assets/figs/badge-website.svg)](https://compvis.github.io/fm-boosting/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2312.07360)


![cover](/assets/figs/cover-fig.png)

**Samples synthesized in $1024^2$ px.** We elevate DMs and similar architectures to a higher-resolution domain, achieving exceptionally rapid processing speeds. We leverage the [Latent Consistency Models (LCM)](https://arxiv.org/abs/2310.04378), distilled from [SD1.5](https://arxiv.org/abs/2112.10752)  and [SDXL](https://arxiv.org/abs/2307.01952), respectively. To achieve the same resolution as LCM (SDXL), we boost LCM-SD1.5 with our general Coupling Flow Matching (CFM) model. This yields a further speedup in the synthesis process and enables the generation of high-resolution images of high fidelity in an average $`0.347`$ seconds. The LCM-SDXL model fails to produce competitive results within this shortened timeframe, highlighting the effectiveness of our approach in achieving both speed and quality in image synthesis.


## 📝 Overview

In this work, we leverage the complementary strengths of Diffusion Models (DMs), Flow Matching models (FMs), and Variational AutoEncoders (VAEs): the diversity of stochastic DMs, the speed of FMs in training and inference stages, and the efficiency of a convolutional decoder to map latents into pixel space. This synergy results in a small diffusion model that excels in generating diverse samples at a low resolution. Flow Matching then takes a direct path from this lower-resolution representation to a higher-resolution latent, which is subsequently translated into a high-resolution image by a convolutional decoder. We achieve competitive high-resolution image synthesis at $1024^2$ and $2048^2$ pixels with minimal computational cost.

## 🚀 Pipeline

During training we feed both a low- and a high-res image through the pre-trained encoder to obtain a low- and a high-res latent code. Our model is trained to regress a vector field which forms a probability path from the low- to the high-res latent within $t \in [0, 1]$.

![training](assets/figs/pipeline-train.jpg)

At inference we can take any diffusion model, generate the low-res latent, and then use our Coupling Flow Matching model to synthesize the higher dimensional latent code. Finally, the pre-trained decoder projects the latent code back to pixel space, resulting in $1024^2$ or $2048^2$ images.

![inference](assets/figs/pipeline-inf.jpg)


## 📈 Results

We show zero-shot quantitative comparison of our method against other state-of-the-art methods on the COCO dataset. Our method achieves a good trade-off between performance and computational cost.

![results-coco](assets/figs/coco-comparison.jpg)

We can cascade our models to increase the resolution of a $128^2$ px LDM 1.5 generation to a $2048^2$ px output.

![cascading](assets/figs/128_to_2k-universe.jpg)

You can find more qualitative results on our [project page](https://compvis.github.io/fm-boosting/).

## 🔥 Usage

###
Please execute the following command to download the first stage autoencoder checkpoint:
```
mkdir checkpoints
wget -O checkpoints/sd_ae.ckpt https://www.dropbox.com/scl/fi/lvfvy7qou05kxfbqz5d42/sd_ae.ckpt?rlkey=fvtu2o48namouu9x3w08olv3o&st=vahu44z5&dl=0
```

### Data
For training the model, you have to provide a config file. An example config can be found in `configs/flow400_64-128/unet-base_psu.yaml`. Please customize the data part to your use case. 

In order to speed up the training process, we pre-computed the latents. Your dataloader should return a batch with the following keys, i.e. `image`, `latent`, and `latent_lowres`. Please notice that we use pixel space upsampling (*PSU* in the paper), therefore the `latent` and `latent_lowres` should have the same spatial resolution (refer to L228 `extract_from_batch()` in `fmboost/trainer.py`). 


### Training

Afterwards, you can start the training with

```bash
python3 train.py --config configs/flow400_64-128/unet-base_psu.yaml --name your-name --use_wandb
```

the flag `--use_wandb` enables logging to WandB. By default, it only logs metrics to a CSV file and tensorboard. All logs are stored in the `logs` folder. You can also define a folder structure for your experiment name, e.g. `logs/exp_name`.

### Resume checkpoint

If you want to resume from a checkpoint, just add the additional parameter

```bash
... --resume_checkpoint path_to_your_checkpoint.ckpt
```

This resumes all states from the checkpoint (i.e. optimizer states). If you want to just load weights in a non-strict manner from some checkpoint, use the `--load_weights` argument.

### Inference
*We will release a pretrained checkpoint and the corresponding inference jupyter notebook soon. Stay tuned!*



## 🎓 Citation

Please cite our paper:

```bibtex
@misc{fischer2023boosting,
      title={Boosting Latent Diffusion with Flow Matching}, 
      author={Johannes S. Fischer and Ming Gui and Pingchuan Ma and Nick Stracke and Stefan A. Baumann and Vincent Tao Hu and Björn Ommer},
      year={2023},
      eprint={2312.07360},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

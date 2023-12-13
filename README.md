<p align="center">
 <h2 align="center">Boosting Latent Diffusion with Flow Matching</h2>
 <p align="center"> 
 Johannes S. Fischer<sup>*</sup> · Ming Gui<sup>*</sup> · Pingchuan Ma<sup>*</sup> · 
 <!-- </p>
  <p align="center">  -->
 Nick Stracke · Stefan A. Baumann · Björn Ommer
 </p>
 <p align="center"> 
CompVis Group, LMU Munich
 </p>
  <p align="center"> <sup>*</sup> <i>denotes equal contribution</i> </p>
</p>

<p align="center">
&rArr; <b>code coming soon!</b>
</p>



## **[Preprint](https://arxiv.org/abs/2312.07360)**, **[Slides](assets/ldm_cfm_slides.pdf)**

## Abstract

Recently, there has been tremendous progress in visual synthesis and the underlying generative models.
Here, diffusion models (DMs) stand out particularly, but lately, flow matching (FM) has also garnered
considerable interest. While DMs excel in providing diverse images, they suffer from long training and
slow generation. With latent diffusion, these issues are only partially alleviated. Conversely, FM offers
faster training and inference but exhibits less diversity in synthesis. We demonstrate that introducing FM between the Diffusion model and the convolutional decoder in Latent Diffusion models offers high-resolution image synthesis with reduced computational cost and model size.  Diffusion can then efficiently provide the necessary generation diversity. FM compensates for the lower resolution, mapping the small latent space to a high-dimensional one.
Subsequently, the convolutional decoder of the LDM maps these latents to high-resolution images. By
combining the diversity of DMs, the efficiency of FMs, and the effectiveness of convolutional decoders, we
achieve state-of-the-art high-resolution image synthesis at $1024^2$ with minimal computational
cost. Importantly, our approach is orthogonal to recent approximation and speed-up strategies for the
underlying DMs, making it easily integrable into various DM frameworks.

![pipeline](https://github.com/CompVis/fm-boosting/blob/main/assets/figs/pipeline_SR.png)


## Results

![frontpage](https://github.com/CompVis/fm-boosting/blob/main/assets/figs/front-page-fig.png)
Samples synthesized in $`1024^2`$ px. We elevate DMs and similar architectures to a higher-resolution domain, achieving exceptionally rapid processing speeds. We leverage the [Latent Consistency Models (LCM)](https://arxiv.org/abs/2310.04378), distilled from [SD1.5](https://arxiv.org/abs/2112.10752)  and [SDXL](https://arxiv.org/abs/2307.01952) respectively. To achieve the same resolution as LCM (SDXL), we boost LCM (SD1.5) with our general Coupling Flow Matching (CFM) model. This yields a further speedup in the synthesis process and enables the generation of high-resolution images of high fidelity in an average $`0.347`$ seconds. The LCM (SDXL) model fails to produce competitive results within this shortened timeframe, highlighting the effectiveness of our approach in achieving both speed and quality in image synthesis.

---

![LHQ](https://github.com/CompVis/fm-boosting/blob/main/assets/figs/LHQ.jpg)

Super-resolution samples from the LHQ dataset. *Left*: low-resolution ground truth image bi-linearly up-sampled. *Right*: high resolution image up-sampled in latent space with our CFM model.

---

![faces_zoom](https://github.com/CompVis/fm-boosting/blob/main/assets/figs/faces-hq-zoom.png)
Up-sampling results with resolution $`1024 \times 1024`$ on the FacesHQ dataset. *Left*: Regression model trained in latent space with the same number of parameters as the flow matching model. *Middle*: Bi-linear up-sampling of the low-resolution image in pixel space. *Right*: Up-sampling in latent space $`32^2 \rightarrow 128^2`$ with our Conditional Flow Matching model and Dormand-Prince ODE solver.


# BMEVITMMA19-Enhancing-Image-Super-Resolution-with-Generative-Adversarial-Networks
BMEVITMMA19 task: SRGAN and ESRGAN implementation

Team members:
- Marco Aiello     - J9PDZZ
- Mend-Amar Badral - HSTV4I
- Thipphsone Phaxy - FQ9TSP

## Project description
This project is a study of the Image Super Resolution (ISR) domain and will implement GAN-based models, particularily SRGAN and ESRGAN in PyTorch framework. The goal is to generate higher resolution images from lower resolution images. Our overarching goal is to compare traditional interpolation methods such as bilinear and bicubic to GAN-based methods. We've taken special importance on implementing the SRGAN network, with comments to help to understand the network architecture.

## Datasets overview
### Div2k
Div2k is a large high quality image dataset specifically for the ISR problem domain. It is introduced in [this](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Agustsson_NTIRE_2017_Challenge_CVPR_2017_paper.pdf) paper and collected to evaluate and benchmark ISR solution submissions for a competition. This dataset is a collection of 1000 2K resolution (meaning pixels on at least one of the axes are 2K) images divided into:
- `train` - 800
- `validation` - 100
- `test` - 100
It provides x2, x3 and x4 downsized images using bicubic and "unknown" methods.
This dataset has been uploaded to `huggingface` library. This simplifies the process of training the model significantly.

### CelebA
CelebA is a large-scale face attributes dataset with more than 200K celebrity images.

### Details on how the `datasets` was downloaded and used
The Div2K data was first uploaded to `huggingface` repository by `@mAiello00` which can be used by `datasets` library to download locally. For more information on how to upload the dataset to huggingface, please check this [link](https://huggingface.co/docs/datasets/en/share). For the notebook, we've provided huggingface token authorization for downloading the dataset. 

For downloading the dataset to local instance (on Colab), we've used:
```python
import os
from huggingface_hub import snapshot_download
dataset_dir = snapshot_download(repo_id="mAiello00/DIV2K", repo_type="dataset")
print("Dataset contains: ", os.listdir(dataset_dir))
high_res_dir = os.path.join(dataset_dir, "DIV2K_train_HR")
low_res_dir  = os.path.join(dataset_dir, "DIV2K_train_LR_bicubic_X2", "DIV2K_train_LR_bicubic", "X2") # Downscaled 2 times images
```
After running a code snippet above, the `huggingface_hub` library would download the dataset and return the `high_res_dir` and `low_res_dir`. For ISR problem, high resolution images correspond to ground truths and low resolution images are inputs to the network.


## Instructions to run the solution
The `jupyter` notebook will be uploaded. To open the notebook in Colab, please change the notebook's link `github.io/your/notebook/link` to `githubtocolab.io/your/notebook/link`. 

Or you could open the notebook via this badge:

[![Open In Colab](https://colab.research.google.com/drive/13amKtIOEObSN3Ce3_FMS2Mia3RFJrLeb?usp=sharing)

## Useful links

1. [SRGAN Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf)

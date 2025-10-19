# BMEVITMMA19 Enhancing Image Super-Resolution with Generative Adversarial Networks(GANs)
BMEVITMMA19 task: SRGAN implementation

# Group Name: Aiello Badral

Team members:
- Marco Aiello     - J9PDZZ
- Mend-Amar Badral - HSTV4I
- Thipphsone Phaxy - FQ9TSP

## 1. Project description
This project is a study of the Image Super Resolution (ISR) domain and will implement GAN-based models, particularily SRGAN (and later ESRGAN) using the PyTorch framework. The architecture to be implemented is outlined in [SRGAN Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf). 

The primary challenge was to implement the Generator, Discriminator, and Residual Block architecture outlined in the SRGAN paper, and then train them.

The goal is to generate higher resolution images from lower resolution images, and compare traditional interpolation methods such as bilinear and bicubic to GAN-based methods.

## 2. Datasets overview
### Div2k
Div2k is a large high quality image dataset specifically for the ISR problem domain. It is introduced in [NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Agustsson_NTIRE_2017_Challenge_CVPR_2017_paper.pdf), and dowloaded from [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/). It is one of the suggested datasets for the assignmet. 

This dataset is a collection of 1000 2K resolution images divided into:
- `train` - 800
- `validation` - 100
- `test` - 100
It provides x2, x3 and x4 downsized images using bicubic and "unknown" methods.

A subset of 400 image pairs from the 'train' set was used in this implementation for training purposes. This was done for two reasons:
1. To avoid Huggingface download limits we ran into while preparing the dataset
2. Reduce the computational resources required for training the models over high epoch counts. This arose from another issue we ran into where Google Colab restricted our GPU useage during one of our earlier tests.

This subset was uploaded to Huggingface and is remotely downloaded when the solution is ran.

### CelebA
CelebA is a large-scale face attributes dataset with more than 200K celebrity images. It was not used for this implementation. This is the other suggested dataset for the assignment. It will be used in our ESRGAN implementation.

### Details on how the `datasets` was downloaded and used
The Div2K data was first uploaded to `huggingface` repository by `@mAiello00` which can be used by `datasets` library to download locally. The huggingface authorization token that is part of the notebook will allow access to this dataset. 

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

## 3. Repository Files

### SRGAN Implementation.ipynb
This file contains the entirety of the solution for the assignment. The function of each cell in the notebook is as follows:

#### Cell(s) 1 - 3
Team member documentation. Required installs. Required imports used throughout the solution.

#### Cell(s) 4
Changes the device to 'cuda' and sets hyperparameters.
Learning rate set to 1e-4 as specified in Section 3.3 of the SRGAN paper.
Batch Size set to 16 for the same reason.
Epoch number was set to 60 due to us running into GPU limitations on Google Colab. Ideally, upwards of 1000 epocks would be preferred.

#### Cell(s) 5-6
Used to load the dataset from Huggingace. As previously mentioned, the dataset we used was uploaded to Huggingface (repository mAiello00/DIV2K). This was done because Google Colab will not save these file between sessions. Saves the High Resolution and Low Resolution datasets.

#### Cell(s) 7
##### ImageDataset class
This class is used to represent the data we train the Generator and Discrimintaor with. It uses the list of high-resolution and low-resolution images, converting them into tensors and taking a random crop of each. A crop size of 96x96 pixels was used. This is done to reduce computational requirements during training as well as improve localized upscaling.

#### Cell(s) 8
Instanciates ImageDataset 

#### Cell(s) 9
##### ResidualBlock class
This class is used to represent the Residual Block architecture described in the papaer. The structure described is k3n64s1

#### Cell(s) 10
##### UpsamplingBlock class
This class is used to represent the Upsampling Block architecture described in the paper. The structure is k3n256s1.

#### Cell(s) 11
##### Generator class
This class is used to represent the Generator architecture described in the paper.

#### Cell(s) 12
##### Discriminator class
This class is used to represent the Discriminator architecture described in the paper.

#### Cell(s) 13
Defines the loss functions used in the training loop. Binary Cross-Entropy (BCE) is used for 'Adversarial Loss'. Mean Square Error (MSE) is used in combination with VGG19 for 'Perceptual Loss.'

#### Cell(s) 14
This is the training loop. It alternates between training the Generator and Discriminator. The Discriminator learns to distinguish between real high-resolution images and those created by the Generator. The Generator learns to create images that appear to be real to the Discriminator. The Generator's loss is measured with MSE and VGG19 feature maps. The result is that the generator slowly learns to create more realistic images.

#### Cell(s) 15
Displays a random set of 3 images. Done so we can see how well the Generator can upscale the images.

## 4. Results

The trained network is able to output following image for super resolution image task, meaning upsampled the low resolution image.

![image](srgan_output.png)


## Instructions to Run the Solution
The `jupyter` notebook contains the solution. Download the file titled 'SRGAN Implementation'. Pressing 'Run' on each of the cells in order (top-to-bottom) will produce the expected results.

## SRGAN Paper

1. [SRGAN Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf)

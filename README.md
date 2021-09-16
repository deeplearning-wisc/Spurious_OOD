
# On the Impact of Spurious Correlation for Out-of-distribution Detection
This codebase provides a Pytorch implementation for the paper: On the Impact of Spurious Correlation for Out-of-distribution Detection. Some parts of the codebase are adapted from [GDRO](https://github.com/kohpangwei/group_DRO).

## Abstract
Modern neural networks can assign high confidence to inputs drawn from outside the training distribution, posing threats to models in real-world deployments. While much research attention has been placed on designing new out-of-distribution (OOD) detection methods, the precise definition of OOD is often left in vagueness and falls short of the desired notion of OOD in reality. In this paper, we present a new formalization and model the data shifts by taking into account both the invariant and environmental (spurious) features. Under such formalization, we systematically investigate how spurious correlation in the training set impacts OOD detection. Our results suggest that the detection performance is severely worsened when the correlation between spurious features and labels is increased in the training set. We further show insights on detection methods that are more effective in reducing the impact of spurious correlation, and  provide theoretical analysis on why reliance on environmental features leads to high OOD detection error.  Our work aims to facilitate better understanding of OOD samples and their formalization, as well as the exploration of methods that enhance OOD detection.

## Illustration
![main](main.png)

## Required Packages
Our experiments are conducted on Ubuntu Linux 20.04 with Python 3.8 and Pytorch 1.6. Besides, the following packages are required to be installed:
* Scipy
* Numpy
* Sklearn
* Pandas

## Datasets
### In-distribution Datasets

- In-distribution training sets:
  - WaterBirds:  Similar to the construction in [Group_DRO](https://github.com/kohpangwei/group_DRO), this dataset is constructed by cropping out birds from photos in the Caltech-UCSD Birds-200-2011 (CUB) dataset (Wah et al., 2011) and transferring them onto backgrounds from the Places dataset (Zhou et al., 2017).
  - [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html): Large-scale CelebFaces Attributes Dataset. The data we used for this task is listed in `datasets/celebA/celebA_split.csv`, and after downloading the dataset, please place the images in the folder of `datasets/celebA/img_align_celeba/`. 
  - ColorMINST:  A colour-biased version of the original [MNIST](http://yann.lecun.com/exdb/mnist/) Dataset. 

### Out-of-distribution Test Datasets

####  Non-spurious OOD Test Sets

Following common practice, we choose three datasets with diverse semantics as non-spurious OOD test sets. We provide links and instructions to download each dataset:
* [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of `datasets/ood_datasets/dtd`.
* [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz): download it and place it in the folder of `datasets/ood_datasets/LSUN_resize`.
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): download it and place it in the folder of `datasets/ood_datasets/iSUN`.
* [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat): download it and place it in the folder of `datasets/ood_datasets/svhn`. 
* Synthetic Images: code for generating a Gaussian noise dataset is contained in `datasets/gaussian_dataset.py`

For example, run the following commands in the **root** directory to download **LSUN-R**:
```
cd datasets/ood_datasets
wget https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz
tar -xvzf LSUN.tar.gz
```

#### Spurious OOD Test Sets
* Color MNIST: can be downloaded [here](https://www.dropbox.com/s/kqqm9doda33f4tt/partial_color_mnist_0%261.zip?dl=0) and place it under `datasets/ood_datasets`.
* WaterBirds: refer to [Waterbirds](#WaterBirds) and the dataset should be placed in the folder of `datasets/ood_datasets/placesbg`.
* CelebA: the meta data for this dataset has already been included in the provided CelebA zip file as `datasets/CelebA/celebA_ood.csv`.


## Quick Start
To run the experiments, you need to first download and place the datasets in the specificed folders as instructed in [Datasets](#Datasets). We provide the following commands and general descriptions for related files.

### ColorMNIST
* `datasets/color_mnist.py` downloads the original MNIST and applies colour biases on images by itself. No extra preparation is needed on the user side.

Here is an example for training on the ColorMNIST Dataset and OOD evaluation:
```bash
python train_bg.py --gpu-ids 0 --in-dataset color_mnist --model resnet18 --epochs 30 --save-epoch 10 --data_label_correlation 0.45 --domain-num 8 --method erm --name erm_r_0_45 --exp-name cdann_r_0_45_2021-08-31
python test_bg.py --gpu-ids 0 --in-dataset color_mnist --model resnet18 --test_epochs 30 --data_label_correlation 0.45 --method cdann --name cdann_r_0_45 --exp-name cdann_r_0_45_2021-08-31
python present_results_py --in-dataset color_mnist --name cdann_r_0_45 --exp-name cdann_r_0_45_2021-08-31 --test_epochs 30
```
Notes for some of the arguments:
* `--data_label_correlation`: the correlation between labels and spurious feature (which is the background color here), as explained in the paper.
* `--method`: selected from 'erm', 'irm', 'gdro', 'rex', 'dann', 'cdann', 'rebias'. The same applies to the experiments below. The uploaded codebase only contains option 'erm'. The complete list of options will be available soon.
* `--name`: by convention, here we specify the name as Method_Correlation. Users are welcome to use other names for convenience.
* `--gpu-ids`: the index of the gpu to be used. Currently we support running with a single gpu. Support for Distributed training will be provided soon.
* `--domain-num`: will be used for others other than 'erm'. This option will be used for domain invariance learning methods.
### WaterBirds
* `datasets/cub_dataset.py`: provides the dataloader for WaterBirds datasets of multiple correlations.
* `datasets/generate_waterbird.py`: generates the combination of bird and background images with a preset correlation. You can simply run `python generate_waterbird.py` to generate the dataset and the dataset will be stored as `datasets/waterbird_completexx_forest2water2`, where `xx` is the string of the two digits after the decimal point, for example when r=0.9, `xx`=90.
* `datasets/generate_placebg.py`: subsamples background images of specific types as the OOD data. You can simply run `python generate_placebg.py` to generate the OOD dataset, and it will be stored as `datasets/ood_datasets/placesbg/`.

(Notes: Before the generation of WaterBirds dataset, you need to download and change the path of CUB dataset and Places dataset first as specified in `generate_waterbird.py`.)

A sample script to run model training and ood evaluation task on WaterBirds is as follows:
```bash
python train_bg.py --gpu-ids 0 --in-dataset waterbird --model resnet18 --epochs 30 --save-epoch 10  --data_label_correlation 0.9 --domain-num 4 --method erm --name erm_r_0_9 --exp-name erm_r_0_9_2021-08-31
python test_bg.py --gpu-ids 0 --in-dataset waterbird --model resnet18 --test_epochs 30 --data_label_correlation 0.9 --method erm --name erm_r_0_9 --exp-name erm_r_0_9_2021-08-31
python present_results_py --in-dataset waterbird --name erm_r_0_9 --exp-name erm_r_0_9_2021-08-31 --test_epochs 30
```
Notes for some of the arguments:
* `--data_label_correlation`: can be selected from [0.5, 0.7, 0.9].

### CelebA
* `datasets/celebA_dataset.py`: provides the dataloader for CelebA datasets and OOD datasets.

A sample script to run model training and ood evaluation task on CelebA is as follows:
```bash
python train_bg.py --gpu-ids 0 --in-dataset celebA --model resnet18 --epochs 30 --save-epoch 10  --data_label_correlation 0.8 --domain-num 4 --method erm --name erm_r_0_8 --exp-name erm_r_0_8_2021-08-31
python test_bg.py --gpu-ids 0 --in-dataset celebA --model resnet18 --test_epochs 30 --data_label_correlation 0.8 --method erm --name erm_r_0_8 --exp-name erm_r_0_8_2021-08-31
python present_results_py --in-dataset waterbird --name erm_r_0_8 --exp-name erm_r_0_8_2021-08-31 --test_epochs 30
```
Notes for some of the arguments:
* `--data_label_correlation`: the correlation for this experiment and can be selected from[0.7, 0.8].

### For bibtex citation

```
@misc{ming2021impact,
      title={On the Impact of Spurious Correlation for Out-of-distribution Detection}, 
      author={Yifei Ming and Hang Yin and Yixuan Li},
      year={2021},
      eprint={2109.05642},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

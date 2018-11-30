# AOD-Net

![](https://img.shields.io/badge/pytorch-0.4.0-blue.svg) ![](https://img.shields.io/badge/python-3.6.5-brightgreen.svg)

## Introducion

This project is a *pytorch implementation* of AOD-Net : All-in-One Network for Dehazing. Recently, there are a number of good implementations:
- [Boyiliee/AOD-Net](https://github.com/Boyiliee/AOD-Net) the original author's project, developed based on pycaffe, only have pre-trained model and inference code
- [TheFairBear/PyTorch-Image-Dehazing](https://github.com/TheFairBear/PyTorch-Image-Dehazing) developed based on pytorch
- [weber0522bb/AODnet-by-pytorch](https://github.com/weber0522bb/AODnet-by-pytorch) developed based on pytorch
- [https://github.com/sachinpuranik99/AOD_Net](https://github.com/sachinpuranik99/AOD_Net) developed based on keras

During our implementing, we referred the above implementations. However, our implementation has several unique and new features compared with the above implementations:
- **It has both train and test code** We provided both train and test code and pre-trained pytorch pickle file
- **It has more clearly code structure** We refactore the code structure of the dataloader to make it more consistent with pytorch dataset
- **It has better logging decorator** We add logging decorator for network pipeline function calling

## Preparation
First of all, clone the code
```
git clone https://github.com/walsvid/AOD-Net.pytorch.git
```
Then, install prerequisites
```
pip install -r requirements.txt
```
### Data Preparation
Please download the `training images` and `original images` from [author's web page](https://sites.google.com/site/boyilics/website-builder/project-page).

Then make a directory for data, change the parameters about data directories.
### Pre-trained model
Please download the pretrained model from [this download link]().
## Train
```
chmod +x run_train.sh
./run_train.sh
```
You can change the parameter in train bash script to satisfied your project.
## Test
```
chmod +x run_test.sh
./run_test.sh
```
## Demo
This is the dehazing result image comparison. Left image is haze image, right image is clean image processed by AOD-Net.
![](https://i.loli.net/2018/11/30/5c00f22dbeb9d.jpg)

## TODO
- [ ] Integrate existing models with detection tasks

## Citation
If you using this project in your work, please don't forget to cite the original author's paper.
```
@inproceedings{li2017aod,
  title={Aod-net: All-in-one dehazing network},
  author={Li, Boyi and Peng, Xiulian and Wang, Zhangyang and Xu, Jizheng and Feng, Dan},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  volume={1},
  number={4},
  pages={7},
  year={2017}
}
```

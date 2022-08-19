PyTorch implementation for paper: Group-wise interactive region learning for zero-shot recognition.

torch ==1.6.

h5py ==3.7.0

imageio == 2.21.0

scipy ==1.7.3

scikit-learn == 1.0.2

scikit-image == 0.17.2

sklearn == 0.0

torchvision == 0.7.0

tqdm == 4.64.0 

wheel == 0.37.1


## Data and Model Preparation

Please download CUB, SUN, AWA2 datasets, and ResNet101 pretrained model.
- Dataset: please download the dataset, i.e., [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [AWA2](https://cvml.ist.ac.at/AwA2/), [SUN](https://groups.csail.mit.edu/vision/SUN/hierarchy.html), and change the arg.data to the dataset root path on your machine

- Pre-trained models: please download the [pre-trained models](https://drive.google.com/file/d/1c5scuU0kZS5a9Rz3kf5T0UweCvOpGsh2/view?usp=sharing) and place it in *./pretrained_models/*.

## Test
Please download the pretrained model (https://drive.google.com/drive/folders/1u3gLKWjr8RfaLb1mA8Wo6T-Aml1JKs-U?usp=sharing).

Please specify the '--data' (for dataset path), '--resnet_pretrain'(ResNet101 pretrained model), '--resume' (pretrained model ), and then run:

cub_test.py
sun_test.py
awa2_test.py



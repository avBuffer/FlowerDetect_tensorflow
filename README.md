# FlowerDetect_tensorflow
How to train and verify flowers by using CNN based on tensorflow?

## Introduction
* Help to study Tensorflow CNN, and realize a sample about flowers data.

## Requirements
* Python
* NumPy
* [Tensorflow](https://github.com/tensorflow/tensorflow)

## Usage
* Scripts
```shell
git clone --recursive https://github.com/avBuffer/FlowerDetect_tensorflow.git
```
* Read get_flower_data.txt and the get flowers data
** Download site: http://download.tensorflow.org/example_images/flower_photos.tgz
** Unzip flower_photos.tgz and move to data folder 
```shell
cd FlowerDetect_tensorflow/data
tar -zxvf flower_photos.tgz 
mv flower_photos data/flowers
```

* Train and detect flowers
```shell
cd FlowerDetect_tensorflow/src
python flower_train.py
python flower_detect.py
```

* Also import the project into Eclipse, such as "File->Import->General/Existing Projects into Workspace". But you should install PyDev in Eclipse.

## Issues
* If you have any idea or issues, please keep me informed.
* My Email: jalymo at 126.com, and my QQ/Wechat: 345238818

## Wechat&QQ group 
* I setup VoAI Wechat group, which discusses AI/DL/ML/NLP.
* VoAI means Voice of AI, Vision of AI, Visualization of AI etc.
* Also you can joint QQ group ID: 183669028

Any comments or issues are also welcomed.Thanks!

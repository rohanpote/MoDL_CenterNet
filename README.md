# Object Detection Project based on CenterNet (Object as Points)

In this project we study the problem of Object Detection as discussed in the reference[1] We use the MS-COCO 2017 data for training.

This project was done as part of our course on Machine Learning for Image Processing (ECE 285) at UCSD.

(Contributors: Aditya Sant, Pulak Sarangi, Rohan R. Pote)

## Dependencies

Please refer to CenterNet Readme for the details of dependencies and full functionality of this project. Below we mention steps to run before using the Demo, TrainingDemo and experiment-specific notebooks in src/local_notebooks folder.


### Git clone the repository
~~~
$ git clone https://github.com/rohanpote/MoDL_CenterNet.git
~~~

### Run the make fies (from current directory)
~~~
$ (cd src/lib/external && make)
~~~
~~~
$ (cd src/lib/models/networks/DCNv2 && ./make.sh)
~~~


### Different Experiments

The src/local_notebooks/ contains the different experiment specific notebooks. Following is the nature of experiments run:

1. Different training set sizes

2. Different loss functions for offset and size regression (tried novel weighted L_1!)

3. Added one (non-learnable for now) additional layer to increase the dimension of the input image. This experiment tried to improve the AP and AR score for small objects. The results satisfactorily improved the desired score, but performed poor for large objects. More research is needed to do simultaneously improve overall mAP.

Please refer to CenterNet Readme to download pre-trained models from the original authors and MS-COCO website for downloading the data.

## Reference

[1] X. Zhou, D. Wang, P. Krahenbuhl,"Objects as Points", in arXiv preprint arXiv:1904.07850, 2019.
[2] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft ´coco: Common objects in context. In ECCV, 2014.
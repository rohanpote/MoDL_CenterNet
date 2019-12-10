# Object Detection Project based on CenterNet (Object as Points)

In this project we study the problem of Object Detection as discussed in the reference[1]

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


## Reference
- <img src="https://latex.codecogs.com/gif.latex?\text{[1] X. Zhou, D. Wang, P. Kr${\"a}$henb${\"u}$hl,"Objects as Points",}\textit{in arXiv preprint arXiv:1904.07850}, 2019."/> 
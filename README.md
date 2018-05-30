# Wirifying App Screenshots
An approach to automatically collect different types of UI components from screenshots of existing mobile apps.

This work leverages a Tensorflow implementation of Faster RCNN, mainly based on the work of [endernewton](https://github.com/endernewton/tf-faster-rcnn).

## Prerequisites
* [Tensorflow r1.2](https://www.tensorflow.org/versions/r1.2/install/)
* [Python 3.6](https://www.python.org/downloads/)
```Shell
pip install Cython opencv-python easydict numpy scipy scikit-image six lxml Pillow imgaug
```

## Getting Started
1. Clone the repository
  ```Shell
  git clone 
  ```

2. Update your -arch in setup script to match your GPU
  ```Shell
  cd FYP_2017/tf-faster-rcnn/lib
  # Change the GPU architecture (-arch) if necessary
  vim setup.py
  ```

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

  **Note**: You are welcome to contribute the settings on your end if you have made the code work properly on other GPUs. Also even if you are only using CPU tensorflow, GPU based code (for NMS) will be used by default, so please set **USE_GPU_NMS False** to get the correct output.


3. Build the Cython modules
  ```Shell
  make clean
  make
  cd ..
  ```

## Test with pre-trained model
1. Download pre-trained model
  - Android dataset with image augmentation [here](https://drive.google.com/open?id=1D324PezWrsS1tpYLIreKQhSOVdHFbwr-).
  ```Shell
  unzip android_aug_res101_140k.zip
  ```

2. Setup to use the pre-trained model
  ```Shell
  sudo cp -r android_aug_res101_140k/output .
  sudo cp -r android_aug_res101_140k/output .
  sudo cp -r android_aug_res101_140k/output .
  ```

  ```Shell
  NET=res101
  TRAIN_IMDB=voc_2007_trainval+voc_2012_trainval
  mkdir -p output/${NET}/${TRAIN_IMDB}
  cd output/${NET}/${TRAIN_IMDB}
  ln -s ../../../data/voc_2007_trainval+voc_2012_trainval ./default
  cd ../../..
  ```

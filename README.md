# Sample-Aware Knowledge Distillation between Monocular Camera-based 3D Object Detectors
![main framework_v0 3](https://github.com/user-attachments/assets/d33e52d9-b7bf-48d2-b2d6-623690f0865f)
SAKD is Knowlede distillation method for Monocular-camera based 3D object detection. This project is an implementation of SAKD.



## Installation

* Python 3.6
* CUDA 11.3
* PyTorch 1.11.0 


Clone the repo.
```
git clone https://github.com/SAKD25/SAKD.git
```


Install requirements
```
apt-get update
apt-get install libboost-all-dev
apt-get install libgl1
pip install -r requirements.txt
```

Then please use the Detectron2 included in this project.
```
python setup.py build develop
```



## Model
Download model from below link
* [MonoRCNN++_SAKD_ResNet18](https://drive.google.com/file/d/1xhHFEk5jpAjinyC_9a98yHQgZdhMkfEL/view?usp=sharing)
* [MonoRCNN++_ResNet50](https://drive.google.com/file/d/1WiPAvhNYNG510hpYseMvLm6Ko2LrmfHI/view?usp=sharing)
* [MonoRCNN++_ResNet18](https://drive.google.com/file/d/1aY2UYUclbXQoZTFjE3k5iCDEnpCYBdct/view?usp=sharing)

Organize the downloaded models as follows:
```
├── projects
│   ├── MonoRCNN
│   │   ├── output
│   │   │   ├── MonoRCNN++_SAKD_ResNet18.pth   
│   │   │   ├── MonoRCNN++_ResNet50.pth
│   │   │   ├── MonoRCNN++_ResNet18.pth
```

## Demo
For visualizing SAKD
```
cd projects/MonoRCNN
./main.py --config-file config/MonoRCNN_KITTI_SAKD_ResNet18_demo.yaml --num-gpus 1 --resume --eval-only
or
./main.py --config-file config/MonoRCNN_KITTI_ResNet18_demo.yaml --num-gpus 1 --resume --eval-only
or
./main.py --config-file config/MonoRCNN_KITTI_ResNet50_demo.yaml --num-gpus 1 --resume --eval-only
```


## Validation

First, you need to setting the KITTI dataset

* [KITTI](projects/KITTI/README.md)

Then, run below python file

```
cd projects/MonoRCNN
./main.py --config-file config/MonoRCNN_KITTI_SAKD_ResNet18.yaml --num-gpus 1 --resume --eval-only
or
./main.py --config-file config/MonoRCNN_KITTI_ResNet18.yaml --num-gpus 1 --resume --eval-only
or
./main.py --config-file config/MonoRCNN_KITTI_ResNet50.yaml --num-gpus 1 --resume --eval-only
```
Set [`VISUALIZE`] yaml file `True` to visualize 3D object detection results (saved in `output/evaluation/test/visualization`).


## Training
Training Code will be available soon.

## License
This project is licensed under CC BY-NC 4.0 license. Redistribution and use of the code for non-commercial purposes should follow this license.
Detectron2 is released under the [Apache 2.0 license](LICENSE).


## Acknowledgement
* [Detectron2](https://github.com/facebookresearch/detectron2)
* [MonoRCNN++](https://github.com/Rock-100/MonoDet)

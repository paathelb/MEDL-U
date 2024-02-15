# MEDL-U
[ICRA 2024] The PyTorch implementation of 'MEDL-U: Uncertainty-aware 3D Automatic Annotation based on Evidential Deep Learning.'

## Data Preparation
The KITTI 3D detection dataset can be downloaded from the official webstie: [link](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

## Train
To train a MTrans with the KITTI dataset. Simply run:
> python train.py --cfg_file configs/MTrans_kitti.yaml

## Pretrained Model


## References
The IoU loss module is borrowed from "https://github.com/lilanxiao/Rotated_IoU". We thank the author for providing a neat implementation of the IoU loss.

This work is based from MTrans ("https://github.com/Cliu2/MTrans"). We thank the authors for their opensourced code. 

# Citation

```
@inproceedings{Paat2023MEDLUU3,
  title={MEDL-U: Uncertainty-aware 3D Automatic Annotation based on Evidential Deep Learning},
  author={Helbert Paat and Qing Lian and Weilong Yao and Tong Zhang},
  booktitle={ICRA},
  year={2024}
}
```

# MEDL-U
[ICRA 2024] The PyTorch implementation of 'MEDL-U: Uncertainty-aware 3D Automatic Annotation based on Evidential Deep Learning.'

![Model Architecture](archi.jpg)

## Abstract
Advancements in deep learning-based 3D object detection necessitate the availability of large-scale datasets. However, this requirement introduces the challenge of manual annotation, which is often both burdensome and time-consuming. To tackle this issue, the literature has seen the emergence of several weakly supervised frameworks for 3D object detection which can automatically generate pseudo labels for unlabeled data. Nevertheless, these generated pseudo labels contain noise and are not as accurate as those labeled by humans. In this paper, we present the first approach that addresses the inherent ambiguities present in pseudo labels by introducing an Evidential Deep Learning (EDL) based uncertainty estimation framework. Specifically, we propose MEDL-U, an EDL framework based on MTrans, which not only generates pseudo labels but also quantifies the associated uncertainties. However, applying EDL to 3D object detection presents three primary challenges: (1) relatively lower pseudolabel quality in comparison to other autolabelers; (2) excessively high evidential uncertainty estimates; and (3) lack of clear interpretability and effective utilization of uncertainties for downstream tasks. We tackle these issues through the introduction of an uncertainty-aware IoU-based loss, an evidence-aware multi-task loss function, and the implementation of a post-processing stage for uncertainty refinement. Our experimental results demonstrate that probabilistic detectors trained using the outputs of MEDL-U surpass deterministic detectors trained using outputs from previous 3D annotators on the KITTI val set for all difficulty levels. Moreover, MEDL-U achieves state-of-the-art results on the KITTI official test set compared to existing 3D automatic annotators.

## Table of Contents
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Pretrained Model](#pretrained-model)
- [Generate Pseudolabels](#generate-pseudolabels)
- [Postprocess Uncertainties](#postprocess-uncertainties)
- [References](#references)
- [Citation](#citation)




## Data Preparation
The KITTI 3D detection dataset can be downloaded from the official website: [link](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

## Training
Modify the configuration file located at `/configs/MTrans_kitti.yaml`.

To train a MTrans with the KITTI dataset, simply run:

```bash
python train.py --cfg_file configs/MTrans_kitti.yaml
```

## Pretrained Model
You can download the pretrained model here: [link](https://drive.google.com/file/d/1-tzkSk0CdMg9B95b-i4eaTcpahCF3EG8/view?usp=sharing)

## Generate Pseudolabels
To generate pseudolabels, make sure that the 'best_model.pt' is saved in the experiment_name/ckpt folder. Simply run:

```bash
python train.py --cfg_file configs/MTrans_kitti_gen_label.yaml
```

## Postprocess Uncertainties
Details on the postprocessing of uncertainties and integration with KITTI infos and dbinfos files can be found here: /small_experiments/glenet_weights_statistics.ipynb

## References
This work is based from MTrans ("https://github.com/Cliu2/MTrans"). We thank the authors for their open-source code. 

## TODO

# Citation
```
@inproceedings{Paat2023MEDLUU3,
  title={MEDL-U: Uncertainty-aware 3D Automatic Annotation based on Evidential Deep Learning},
  author={Helbert Paat and Qing Lian and Weilong Yao and Tong Zhang},
  booktitle={ICRA},
  year={2024}
}
```

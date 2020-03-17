# Reproducing 3rd Place Solution for Understanding Clouds
The originally solution is located [here](https://github.com/naivelamb/kaggle-cloud-organization).
Xuan Cao's transcript for his solution is located [here](https://www.kaggle.com/c/understanding_cloud_organization/discussion/117949).

This repository is just me practicing seeing if I can match whatever the solution did.

## Workflow
### Preprocessing
Assuming that you've already downloaded the dataset from Kaggle:
```
!python /content/reproducing-cloud/scripts/create_resized_dset.py  --yml_path="/content/understanding-clouds-kaggle/configs/create_dset.yml"
```
Feel free to change the parameters in the .yml file (such as the image sizes).

## Plan
* MVP Cascade [`0.652 Public LB`]
  * ResNet34 + FPN w/ BCE [`0.608 Public LB`]
  * ResNet34 + FPN w/ pos-only soft dice loss
  * Single Fold, no TTA, no classifier, no threshold adjustment (=0.5)
  * This is already better than my previous ensemble of four models that achieved `~0.660 Public LB`
    * And those were with TTA and a classifier!
  * Parameter Summary:
  ```
  Network: Resnet34-FPN
  Image size: 384x576
  Batch size: 16
  Optimizer: Adam
  Scheduler: reduceLR for seg1, warmRestart for seg2.
  Augmentations: H/V flip, ShiftScalerRotate and GridDistortion
  TTA: raw, Horizontal Flip, Vertical Flip
  ```
  * Results:
  ```
  1-fold: 0.664
  5-fold + TTA3: 0.669
  5-fold + TTA3 + classifier: 0.670.
  ```

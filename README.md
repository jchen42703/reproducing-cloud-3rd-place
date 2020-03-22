# Reproducing 3rd Place Solution for Understanding Clouds
The originally solution is located [here](https://github.com/naivelamb/kaggle-cloud-organization).
Xuan Cao's transcript for his solution is located [here](https://www.kaggle.com/c/understanding_cloud_organization/discussion/117949).

This repository is just me practicing seeing if I can match whatever the solution did.

## Workflow
### Preprocessing
Assuming that you've already downloaded the dataset from Kaggle:
```
python /content/reproducing-cloud-3rd-place/scripts/create_resized_dset.py  --yml_path="/content/understanding-clouds-kaggle/configs/create_dset.yml"
```
Feel free to change the parameters in the .yml file (such as the image sizes).
### Training
For stage 1:
```
python /content/reproducing-cloud-3rd-place/scripts/train_yaml.py --yml_path="train_seg1.yml"
```
For stage 2:
```
python /content/reproducing-cloud-3rd-place/scripts/train_yaml.py --yml_path="train_seg2.yml"
```
### Inference
For stage 1 and 2's submission:
```
python /content/reproducing-cloud-3rd-place/scripts/pred_create_sub.py --yml_path="create_sub.yml"
```
Just make sure to specify the proper weights for each stage.

To cascade:
```
WIP
```

## Plan
* MVP Cascade
  * Each network only takes about 2.5 hours to train (with Google Colab P4) (5 min/epoch * 30 epochs)
  * ResNet34 + FPN w/ BCE
    * [Weights](https://drive.google.com/open?id=1ibc0aNyQxxNvPqix9CABAKAAH5p6iL4d) [`0.60813 Public`/ `0.60287 Private LB`]
  * ResNet34 + FPN w/ pos-only soft dice loss
    * [Weights](https://drive.google.com/open?id=1sIYsZQAnfdyykArCvVEIR0VCOszMSJHw) [`0.34752 Public/ 0.35233 Private`]
    * With cascade: [`0.65658 Public/ 0.65195 Private`]
      * This is already better than my previous private LB of `0.650` (and that was with an ensemble of deeper models and TTA...)
        * Goes to show how important keeping aspect ratio in your input images is.
  * Single Fold, no TTA, no classifier, no threshold adjustment (=0.5)
  * Parameter Summary:
  ```
  Network: Resnet34-FPN
  Image size: 384x576
  Batch size: 16
  Optimizer: Adam
  Scheduler: warmRestart for seg1, warmRestart for seg2.
  Augmentations: H/V flip, ShiftScalerRotate and GridDistortion
  TTA: raw, Horizontal Flip, Vertical Flip
  ```

  * Results:
  ```
  1-fold: 0.664
  5-fold + TTA3: 0.669
  5-fold + TTA3 + classifier: 0.670.
  ```

### Training
* WarmRestarts
  * WarmRestart(T_mult=2) for 28 epochs
  * for epochs 29-31, LR = 1e-5
  * for epochs 32-35, LR = 5e-6

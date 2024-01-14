# EMCAM-Net
The codes for the work "EMCAM-Net：Efficient Multi-scale Context Aggregation Mixture Network for Medical Image Segmentation" 

We updated the Reproducibility. I hope this will help you to reproduce the results.

## 1. Method
* Overall architecture of the proposed EMCAM-Net. (b) and (c) respectively represent the Efficient Multi-scale Context Aggregation (EMCA) and Modeling Global Representation (MGR) modules.

  ![image](https://github.com/AloneIsland/EMCAM-Net/blob/master/tool/EMCAM-Net-visio-2.jpg)

## 2. Environment

- Please prepare an environment with python=3.9, and then use the command "pip install -r requirements.txt" for the dependencies.

## 3. Train/Test

- Run the train script on synapse dataset. The batch size we used is 8. 

- Train

```bash
python train.py --dataset Synapse --root_path your DATA_DIR --max_epochs 300 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 8
```

- Test 

```bash
python test.py --dataset Synapse --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR
```

## Reproducibility

- Questions about Dataset

Many of you have asked me for datasets, and I personally would be very glad to share the preprocessed Synapse and ACDC datasets with you. However, I am not the owner of these two preprocessed datasets. Please email jienengchen01 AT gmail.com to get the processed datasets.

- Codes

Regarding how to reproduce the segmentation results presented in the paper, we discovered that different GPU types would generate different results. In our code, we carefully set the random seed, so the results should be consistent when trained multiple times on the same type of GPU. If the training does not give the same segmentation results as in the paper, it is recommended to adjust the learning rate. 

## References
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
* [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet)

## Citation

```bibtex
None
```

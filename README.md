# Kaggle-ISIC2024-69th-solution

Here is the official competition address ---> [ISIC 2024 - Skin Cancer Detection with 3D-TBP](https://www.kaggle.com/competitions/isic-2024-challenge)

Here is the link to the solution we published on Kaggle ---> [69th Place Solution](https://www.kaggle.com/competitions/isic-2024-challenge/discussion/532784)

# 69th Place Solution

First of all, we would like to thank Kaggle and ISIC for organizing this competition, and also express our respect to the participants of this competition.

## Code
The two notebooks we submitted:
1. [https://www.kaggle.com/code/zhiyue666/isic-2024-final-infer-ensemble?scriptVersionId=195307326](https://www.kaggle.com/code/zhiyue666/isic-2024-final-infer-ensemble?scriptVersionId=195307326)(cv:0.182496  PublicLB:0.18440  PrivateLB:0.16941)(lgb_xgb_cb_with_nnFeature)
2. [https://www.kaggle.com/code/xiyan123/isic2024?scriptVersionId=195377751](https://www.kaggle.com/code/xiyan123/isic2024?scriptVersionId=195377751)(cv:0.1732  PublicLB:0.18304  PrivateLB:0.16217)(lgb_xgb_cb_with_onlyTabular Data)
We noticed that after adding nn features to the table features, the importance of nn features far exceeded other features. In order to prevent excessive deviation, we also submitted a notebook that only uses table data.

Here is the notebook for our training: \
train nn Model:[https://www.kaggle.com/code/zhiyue666/isic-2024-dl-model-training](https://www.kaggle.com/code/zhiyue666/isic-2024-dl-model-training) \
train nn ensemble Model:[https://www.kaggle.com/code/zhiyue666/isic-2024-nn-ensemble-training](https://www.kaggle.com/code/zhiyue666/isic-2024-nn-ensemble-training) \
train nnFeature to GBDT:[https://www.kaggle.com/code/zhiyue666/isic-2024-nnfeature-to-gbdt-training](https://www.kaggle.com/code/zhiyue666/isic-2024-nnfeature-to-gbdt-training)

**== You can access the above notebook links directly in Kaggle, or in the src folder of this repository ==**

## DL Part:
In terms of the selection of training sets, we selected 584 positive samples of ISIC2020 and 393 positive samples of ISIC2024 to form 977 positive samples, and downsampled the negative samples of ISIC2024 to 1:1; 1.5:1; 2:1 ratios with positive samples, that is, the total training samples are 1954, 2442, and 2931. Through our experiments, we found that the best effect is obtained when the number of positive and negative samples is 1:1. In addition, we removed these 3 images from the training set: ISIC_0573025, ISIC_1443812, and ISIC_5374420 from [https://www.kaggle.com/competitions/isic-2024-challenge/discussion/521145](https://www.kaggle.com/competitions/isic-2024-challenge/discussion/521145). Thanks to @bobfromjapan and @itsuki9180 for their contributions.

Our validation set selects all 401,059 images of ISIC2024.

For both training and validation, we use StratifiedGroupKFold with patient_id as group. For the training set and validation set, the same isic_id and patient_id have the same kfold. For some patient_ids in ISIC2020 that do not appear in ISIC2024, we simply distribute them equally to each fold.

In terms of training, we tried different batch_size and lr combinations for training. Finally, we selected batch_size=32, lr=1e-3 based on CNN network, which had better results. Based on vit network, we selected batch_size=64, lr=1e-3, which had the best results. However, in our experiments, we found that the effect of vit was always inferior to CNN. We think this may be related to the small image resolution or the fact that we only used some small basic vit. We trained at 96, 128, 160, and 192 resolutions. On the network we used, the resolution of 160 was the best. All subsequent experiments were conducted at the resolution of 160.

The data augmentation we use is the [#1th solution for reference ISIC2020](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175412) and ‘Hair’ augmentation.


```python
class HairAugmentation(A.ImageOnlyTransform):
    def __init__(self, num_hairs_range=(5, 15), hair_color_range=((0, 0, 0), (255, 255, 255)), 
                           always_apply=False, p=0.5):
        super(HairAugmentation, self).__init__(always_apply, p)
        self.num_hairs_range = num_hairs_range
        self.hair_color_range = hair_color_range

    def apply(self, img, **params):
        img = img.copy()
        h, w, _ = img.shape

        num_hairs = random.randint(self.num_hairs_range[0], self.num_hairs_range[1])
        hair_color = (
            random.randint(self.hair_color_range[0][0], self.hair_color_range[1][0]),
            random.randint(self.hair_color_range[0][1], self.hair_color_range[1][1]),
            random.randint(self.hair_color_range[0][2], self.hair_color_range[1][2])
        )

        for _ in range(num_hairs):
            # Randomly choose the position and size of the hair
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(0, w), random.randint(0, h)
            thickness = random.randint(1, 1)  # Making the hair thinner
            img = cv2.line(img, (x1, y1), (x2, y2), hair_color, thickness)

        return img

    def get_params_dependent_on_targets(self, params):
        return {}

    def get_transform_init_args_names(self):
        return ("num_hairs_range", "hair_color_range")

def transform_train(img):
    composition = A.Compose([
        HairAugmentation(num_hairs_range=(5, 15), hair_color_range=((0, 0, 0), (255, 255, 255)), p=0.5),
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightness(limit=0.2, p=0.75),
        A.RandomContrast(limit=0.2, p=0.75),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.),
            A.ElasticTransform(alpha=3),
        ], p=0.7),

        A.CLAHE(clip_limit=4.0, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        A.Resize(CONFIG.img_size[0], CONFIG.img_size[1]),
        A.Cutout(max_h_size=int(CONFIG.img_size[0] * 0.375), max_w_size=int(CONFIG.img_size[1] * 0.375), num_holes=1, p=0.7),    
        A.Normalize(),
        ToTensorV2(),
    ])
    return composition(image=img)["image"]


def transform_val(img):
    composition = A.Compose([
        A.Resize(CONFIG.img_size[0], CONFIG.img_size[1]),
        A.Normalize(),
        ToTensorV2(),
    ])
    return composition(image=img)["image"]
```

We always choose BCELoss over CCELoss because CCELoss performs poorly in our experiments. We also use label smoothing with a threshold of 0.05. We do not use GeMPool in training because it does not perform well in training.

Our final solution is based on 4 models, all of which are directly accessible in timm:
1. tf_efficientnetv2_s.in21k_ft_in1k
2. edgenext_base.in21k_ft_in1k
3. convnext_atto_ols.a2_in1k
4. tf_efficientnet_b3.ns_jft_in1k

Other training configurations:
- n_folds: 5
- seed: 308
- epochs: 32 / fold
- train_batch_size = 32
- valid_batch_size = 512
- img_size = [160, 160]
- learning_rate = 1e-3
- min_lr = 1e-6
- weight_decay = 1e-6
- scheduler: "CosineAnnealingWithWarmupLR"
- optimizer: AdamW

```python

class CosineAnnealingWithWarmupLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, warmup_epochs=10, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.cosine_epochs = T_max - warmup_epochs
        super(CosineAnnealingWithWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [(base_lr * (self.last_epoch + 1) / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            cosine_epoch = self.last_epoch - self.warmup_epochs
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * cosine_epoch / self.cosine_epochs)) / 2 for base_lr in self.base_lrs]
```

A learnable fusion model:
```python

class ensemblelinear(nn.Module):
    def __init__(self, in_features, out_features=1) -> None:
        super().__init__()
        self.model = nn.Linear(in_features, in_features, bias=False)
        self.softmax = nn.Softmax()
        self.out_features = out_features

    def forward(self, x):
        Identity = x
        _tmp = self.model(x)
        _tmp = self.softmax(_tmp)
        _tmp = Identity * _tmp
        output = _tmp.sum(dim=1, keepdim=True)
        return output
```

Finally, we use this fusion model to fuse the outputs of the four NNs into one feature and pass it to the ML Model.

# ML Part
GBDT Model: \
This solution is an improvement based on this notebook \
[https://www.kaggle.com/code/vyacheslavbolotin/isic-2024-only-tabular-data-new-features](https://www.kaggle.com/code/vyacheslavbolotin/isic-2024-only-tabular-data-new-features) \
In terms of model: \
We also used xgb, lgb and cat models for fusion. Although the parameters did not change too much, we merged the models trained from six different seeds(The notebook with only tabular data merges 6 seeds, and the notebook with nnFeature merges 5 seeds.) again in the end, making the results of the models more stable()

In feature engineering: \
Raw numerical feature num-cols \
The numerical feature new-num-cols is added \
Category Feature cat-cols \
Statistical patient case number characteristics special-cols \
The numerical feature norm-cols normalized to 'patient-id'

In addition to the above features in the original notebook, we also calculated the features that normalized the 'patient-id' and 'tbp-lv-location' groups at the same time, which we believe can enable the model to learn information about different parts of different patients, so as to better classify the results.

## The useless part or the part we don't use:
1. TTA: For the inference phase of the nn model, we used 4×TTA and 8×TTA, but the results were not good.
2. DiffuseMix data enhancement: We think this is a good way to expand the data, but we did not try it.
3. MixUp: We also did not use.
4. More Datasets: We tried adding datasets before ISIC2020 to the training, but the results were not good.

## We also spent a lot of time checking whether the data was leaked. We found data leaks in the high-scoring notebooks in the code area, so the CV reference we obtained is not very meaningful.

# The above is our summary of this competition. Thank you for reading.
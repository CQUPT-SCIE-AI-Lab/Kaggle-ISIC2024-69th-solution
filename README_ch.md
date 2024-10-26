# Kaggle-ISIC2024-第69名-解决方案

比赛的官方地址 ---> [ISIC 2024 - Skin Cancer Detection with 3D-TBP](https://www.kaggle.com/competitions/isic-2024-challenge)

以下是我们在 Kaggle 上发布的解决方案的链接 ---> [69th Place Solution](https://www.kaggle.com/competitions/isic-2024-challenge/discussion/532784)

# 第69名 解决方案

首先我们感谢Kaggle与ISIC举办本次比赛，同时也对本次比赛的参赛者表示敬意。

## 代码
我们提交的两个notebook:
1. [https://www.kaggle.com/code/zhiyue666/isic-2024-final-infer-ensemble?scriptVersionId=195307326](https://www.kaggle.com/code/zhiyue666/isic-2024-final-infer-ensemble?scriptVersionId=195307326)(本地cv:0.182496  公榜:0.18440  私榜:0.16941)(lgb_xgb_cb_with_nnFeature)
2. [https://www.kaggle.com/code/xiyan123/isic2024?scriptVersionId=195377751](https://www.kaggle.com/code/xiyan123/isic2024?scriptVersionId=195377751)(本地cv:0.1732  公榜:0.18304  私榜:0.16217)(lgb_xgb_cb_with_onlyTabular Data)
我们注意到在表格特征中加入nn特征之后，nn特征的重要性远超其他特征，为了防止偏差过大，我们在提交的两个notebook中选择提交了一个只使用表格数据的notebook。

这是我们的训练notebook:
训练 nn 模型:[https://www.kaggle.com/code/zhiyue666/isic-2024-dl-model-training](https://www.kaggle.com/code/zhiyue666/isic-2024-dl-model-training) \
训练融合模型:[https://www.kaggle.com/code/zhiyue666/isic-2024-nn-ensemble-training](https://www.kaggle.com/code/zhiyue666/isic-2024-nn-ensemble-training) \
将 nn特征 加入到 GBDT中 训练:[https://www.kaggle.com/code/zhiyue666/isic-2024-nnfeature-to-gbdt-training](https://www.kaggle.com/code/zhiyue666/isic-2024-nnfeature-to-gbdt-training)

**== 您可以直接在 Kaggle 中访问上述笔记本链接，也可以在此存储库的 src 文件夹中访问 ==**

## 深度学习 部分:
在训练集选取方面，选取ISIC2020正样本584个，ISIC2024正样本393个，形成977个正样本, 并将 ISIC2024 的负样本下采样为与正样本的比例为 1:1、1.5:1、2:1, 即总训练样本数分别为1954、2442、2931。通过实验我们发现，当正负样本数为1:1时，效果最好. 此外，我们从训练集中删除了以下 3 幅图像：ISIC_0573025、ISIC_1443812 和 ISIC_5374420 来自 [https://www.kaggle.com/competitions/isic-2024-challenge/discussion/521145](https://www.kaggle.com/competitions/isic-2024-challenge/discussion/521145). 感谢@bobfromjapan 和 @itsuki9180 的贡献。

我们的验证集选择了 ISIC2024 的全部 401,059 张图像。

对于训练和验证，我们使用 StratifiedGroupKFold 并以patient_id 作为组。对于训练集和验证集，相同的 isic_id 和patient_id 具有相同的 kfold。对于 ISIC2020 中某些未出现在 ISIC2024 中的patient_id，我们只是将它们平均分配给每个fold。

训练方面，我们尝试了不同的batch_size和lr组合进行训练，最终基于CNN网络选择了batch_size=32，lr=1e-3，效果最好；基于vit网络选择了batch_size=64，lr=1e-3，效果最好。但是我们在实验中发现vit的效果始终不如CNN，我们认为这可能和图像分辨率较小或者我们只使用了一些较小的基础vit模型有关。我们在96、128、160、192分辨率下进行了训练，在我们使用的网络上，160的分辨率效果最好，后续所有实验均在160分辨率下进行。

我们使用的数据增强是参考 ISIC2020 的 [第1名解决方案](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175412) and ‘头发’ 增强。


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

我们一直选用 BCELoss 而不是 CCELoss，因为 CCELoss 在我们的实验中效果不好。我们还使用阈值为 0.05 的标签平滑。我们没有在训练中使用 GeMPool，因为它在训练中表现出的效果也不好。

我们的最终解决方案基于 4 个模型，所有这些模型都可以在 timm 中直接访问:
1. tf_efficientnetv2_s.in21k_ft_in1k
2. edgenext_base.in21k_ft_in1k
3. convnext_atto_ols.a2_in1k
4. tf_efficientnet_b3.ns_jft_in1k

其他训练配置:
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

可学习的融合模型:
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

最后，我们使用这个融合模型将四个 NN 的输出融合为一个特征，并将其传递给 机器学习 模型。

# 机器学习 部分
GBDT 模型:
此解决方案是基于此笔记本的改进 \
[https://www.kaggle.com/code/vyacheslavbolotin/isic-2024-only-tabular-data-new-features](https://www.kaggle.com/code/vyacheslavbolotin/isic-2024-only-tabular-data-new-features) \
从模型上看:
我们还分别用xgb、lgb和cat模型进行了融合，虽然参数没有太大变化，但最后我们再次合并了从6个不同的种子中训练出来的模型（纯表格数据的notebook合并了6个种子，有nnFeature的notebook合并了5个种子），使得模型的结果更加稳定()

在特征工程中: \
原始数值特征 num-cols \
添加了数值特征 new-num-cols \
类别 特征 cat-cols \
统计病人病例数特征special-cols \
数值特征norm-cols归一化为 'patient-id'

除了原始笔记本中的上述特征外，我们还同时计算了对“patient-id”和“tbp-lv-location”组进行归一化的特征，我们认为这可以使模型学习到不同患者不同部位的信息，从而更好地对结果进行分类。

## 无用的部分或我们没有使用的部分:
1. TTA: 对于 nn 模型的推理阶段，我们使用了 4×TTA 和 8×TTA，但得到的效果都不好。
2. DiffuseMix data enhancement: 我们认为这是扩展数据的好方法，但是我们没有尝试。
3. MixUp: 我们也没有使用。
4. More Datasets: 我们尝试将 ISIC2020 之前的数据集添加到训练中，但得到的效果都不好。

## 我们也花了很多时间检查数据是否泄露。我们在代码区高分notebook中发现了数据泄露，所以其notebook获得的 CV 参考意义不大。

# 以上就是我们对本次比赛的总结，感谢阅读。
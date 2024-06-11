# AICUP2024-TEAM_5218 
   
## Datasets preparation   
### AICUP    
### Visdrone dataset ([https://github.com/VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset) )   
Download task4 VisDrone-MOT dataset (only train dataset)   
> Visdrone dataset lables convert   

Use script visdrone2yolo.py   
```
python visdrone2yolo.py --data_path ./VisDrone-MOT-dataset
```
## Data augmentation   
### Deblur image using NAFNET   
```
python 3.9.5
pytorch 1.11.0
cuda 11.3
```
```
git clone https://github.com/megvii-research/NAFNet.git 
cd NAFNet
pip install -r requirements.txt
python setup.py develop --no_cuda_ext

```
We directly use the pre-trained model **NAFNet-REDS-width64** from the [NAFNet](https://github.com/megvii-research/NAFNet) official Github repo.   
Download the pre-trained model from [NAFNet-REDS-width64.pth](https://drive.google.com/file/d/14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X/view) and add to the folder `NAFNet/experiments/pretrained\_models/`   
Add infermany.py in NAFNET file.   
Modify infermany.py to fit input output path u want. (We used it on all the datasets.)   
```
python infermany.py -opt options/test/REDS/NAFNet-width64.yml
```
### Convert to day-light images using [GSAD](https://github.com/jinnh/GSAD)   
```
git clone https://github.com/jinnh/GSAD.git
```
Installation is same as its github.   
We only use this method on night images in AICUP dataset.   
### Convert to Rainy-like images using [ImgAug](https://github.com/aleju/imgaug)   
Converted by CBX collegue.   
### Combine all the AICUP data in one file : train   
Last word B means rainy-like images, G means day-light images. (Night images with B  means B+G)   
```
train
	--images
		--0902_150000_151900
		--0902_150000_151900B
		--0902_190000_191900
		--0902_190000_191900B
		--0902_190000_191900G
		--...
	--labels
		--0902_150000_151900
		--0902_150000_151900B
		--0902_190000_191900
		--0902_190000_191900B
		--0902_190000_191900G
		--...
```
### Combine all the visdrone data in one file : trainvis   
Last word B means rainy-like images.   
```
trainvis
	--images
		--uav0000013_00000_v
		--uav0000013_00000_vB
		--...
	--labels
		--uav0000013_00000_v
		--uav0000013_00000_vB
		--...

```
## The rest install, data preparation steps was same as [AICUP Baseline: BoT-SORT](https://github.com/ricky-696/AICUP_Baseline_BoT-SORT/tree/main)   
## Fast-ReID model modified   
1. Find path `aicup/fast_reid/fastreid/layers` add `cbam.py`   
2. Modify `__init__.py`, add `from .cbam import CBAM` at the last row.   
3. Find path aicup/fast_reid/fastreid/modeling/backbones/resnet.py and modify.   
   
```
...
from fast_reid.fastreid.layers import (
    IBN,
    SELayer,
    Non_local,
    get_norm,
    CBAM,       
)
...

class ResNet(nn.Module):
    def __init__(self, last_stride, bn_norm, with_ibn, with_se, with_nl, block, layers, non_layers):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = get_norm(bn_norm, 64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, with_ibn, with_se)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, with_ibn, with_se)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, with_ibn, with_se)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm, with_se=with_se)
        self.cbam = CBAM(channel_in=512*block.expansion)
        self.random_init()

        # fmt: off
        if with_nl: self._build_nonlocal(layers, non_layers, bn_norm)
        else:       self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []
        # fmt: on

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm="BN", with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                get_norm(bn_norm, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se))

        return nn.Sequential(*layers)

    def _build_nonlocal(self, layers, non_layers, bn_norm):
        self.NL_1 = nn.ModuleList(
            [Non_local(256, bn_norm) for _ in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512, bn_norm) for _ in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024, bn_norm) for _ in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048, bn_norm) for _ in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # layer 1
        NL1_counter = 0
        if len(self.NL_1_idx) == 0:
            self.NL_1_idx = [-1]
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1

        # layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0:
            self.NL_2_idx = [-1]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1

        # layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0:
            self.NL_3_idx = [-1]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1

        # layer 4
        NL4_counter = 0
        if len(self.NL_4_idx) == 0:
            self.NL_4_idx = [-1]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1
             
        x = x + self.cbam(x)
        return x
...
```
### Train for 60 epoch   
```
python3 fast_reid/tools/train_net.py --config-file fast_reid/configs/AICUP/bagtricks_R50-ibn.yml MODEL.DEVICE "cuda:0"
```
- Best epoch 54   
   
## Train yolov7-w6, yolov7-d6   
### Batch size : 8, image size : 1280   
- yolov7-w6 Best epoch 17   
   
### Batch size : 4, image size : 1280   
- yolov7-d6 Best epoch 16   
   
## Inference with 2 models   
```
bash tools/track_all_timestamps.sh --weights "yolov7-d6/best.pt" "yolov7-w6/best.pt" --source-dir "AI_CUP_MCMOT_dataset/train/images" --device "0" --fast-reid-config "fast_reid/configs/AICUP/bagtricks_R50-ibn.yml" --fast-reid-weights "logs/AICUP/bagtricks_R50-ibn/model_0054.pth"
```
## Result
![image](https://github.com/Plmoleonely/AICUP2024-TEAM_5218/blob/main/image.png)

# AWB-Framework
A framework for training and testing CNN-based AWB methods 

环境在environment.yaml

# 模型训练说明

整体训练逻辑参考`train.py`

## Step1: 模型设置
**模型文件添加**
* 模型主函数单独一个python文件，以其名字命名（e.g. `MobileIE.py` ）；支撑模块函数统一一个python文件， 
以模型名字_modules命名（e.g.`MobileIE_modules.py`）,都放置在./model路径下

**模型文件注册** 
* 在模型主函数（例如`MobileIE.py`）中添加注册函数，同时在类前对其注册，示例如下：
```bash
from registry import register_model
"在class之前注册"
@register_model('MobileIE')
class MobileIENet(nn.Module):
```
* 在`model/__init__.py`和`train.py`中import模型主函数，示例如下：
```bash
"__init__.py"
from .MobileIE import MobileIENet
"train.py"
from models import MobileIE
```

* 在`configs.yaml`中添加模型定义所需参数，示例如下：
```bash
model:
  name: "MobileIE"    ##根据调用的模型修改
  ...
  MobileIE:           ##修改所需参数
    rep_scale: 4
    channels: 12
```

## Step2: 数据集设置
### 光照估计任务
**数据集**
* 单光，2相机：[Gehler's Raw Color Checker Dataset](https://www2.cs.sfu.ca/~colour/data/shi_gehler/) --> `/dataset/CC_full` (已预处理)
* 单光，8相机：[NUS-dataset](http://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html) --> `/dataset/NUS_full` (已预处理)
* 单光，1相机：[CUBE-dataset](https://ipg.fer.hr/ipg/resources/color_constancy) --> `/dataset/cube_raw` (已预处理)

**数据格式示例**
```
dataset_root/
├──img & ground_truth & camera & cc mask
   └── img: *.npy 
   └── ground_truth: *_gt.npy
   └── camera: *_camera.npy
   └── cc mask: *_mask.npy
```


**数据集设置**
* 在`configs.yaml`中修改所用数据集名称，并定义所需参数，示例如下：
```bash
dataset:
  name: "NUS"     ## 所用数据集名称，目前可选：'NUS' or 'ColorChecker' or 'Cube'
  ...
  NUS:            ##修改所需参数
    path: '/dataset/NUS_full'
    folds: 'exclude'
    camera_name: 'SonyA57'
    supported_cameras: False
```

* 在`config.py`中修改训练和测试的batchsize以及worker num， 示例如下：
```bash
parser.add_argument("--train_bz", type=int, default=4)
parser.add_argument("--test_bz", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=9)
```

* 在`train.py`中调用数据集，示例如下：
```bash
from registry import create_dataloaders
from config import parse_args

opt = parse_args()
dataset_name = opt.config['dataset']['name']
dataset_config = opt.config['dataset'][dataset_name]
train_loader, test_loader = create_dataloaders(dataset_name=dataset_name,dataset_config=dataset_config, opt=opt)
```



## Step3: Loss设置
* 一般情况下，光照估计任务默认使用角度误差进行监督（`loss/Loss.py/LossIllum()`），其它任务按照其特点再自行在`Loss.py`内添加，范式示例如下：
```bash
class XXXLoss(Loss):
    def __init__(self):
        super().__init__()

    def _compute(self, pred: Tensor, label: Tensor, safe_v: float = 0.999999) -> Tensor:
        
        return 
```

* 定义loss后，在`train.py`中import，并调用，示例如下：
```bash
from loss import XXXLoss
"Trainer中，训练前"
loss = XXXLoss()
```

## Step4: 优化器设置
`train.py`中默认的是Adam+CosineAnnealingWarmRestarts，如下。可根据训练的模型不同自行修改，
同时在`config.py`中添加或者修改参数：
```bash
"Trainer"
self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    self.optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.min_lr)
```
```bash
"config.py，根据设置添加或者修改参数"
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_period', type=int, default=10)
parser.add_argument('--betas', type=float, default=BETAS)
parser.add_argument('--eps', type=float, default=EPS)
```

## Step4: Trainer和Tester设置

`train.py`中已有基础的训练框架，最好是在该框架下进行模型的训练和框架微调
如有特殊需求，最好重建一个`train_XX.py`文件进行训练

`Test.py`中已有基础的测试框架，最好是在该框架下进行模型的测试和框架微调
如有特殊需求，最好重建一个`Test_XX.py`文件进行测试




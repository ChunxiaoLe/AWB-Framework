import yaml
import torch
from torch.utils.data import DataLoader

"定义模型"
from registry import build_model, build_dataloaders
from Evaluator import Evaluator
"定义loss"
from loss.Loss import LossIllum
from Logger import Logger
from EarlyStopping import EarlyStopping
from config import parse_args
from model.MobileIE import MobileIENet


class Tester:
    def __init__(self):
        self.args = parse_args()
        self.device = self.args.device
        self.logger = Logger(self.args)

        # 构建模型并加载参数
        model_name = self.args.config['model']['name']
        model_cfg = self.args.config['model'].get(model_name, {})
        self.model = build_model(model_name, **model_cfg).to(self.device)
        self.model.load_state_dict(torch.load(self.args.test_model_para), strict=True)

        # 损失
        self.loss = LossIllum()
        self.evaluator = Evaluator()

        # 数据加载
        dataset_name = self.args.config['dataset']['name']
        dataset_config = self.args.config['dataset'][dataset_name]
        _, self.val_loader = build_dataloaders(dataset_name=dataset_name, dataset_config=dataset_config,opt=self.args)


    def eval(self):
        self.model.eval()
        error = []
        with torch.no_grad():
            for data in self.val_loader:
                img, img_gt, label, file_name = data
                img, label = img.to(self.device), label.to(self.device)
                out = self.model(img)
                loss = self.loss(out, label)
                self.evaluator.add_error(loss)
                self.logger.info(f"Img: {file_name} - AE error: {loss:.4f}")

        print("\n *** AVERAGE ACROSS FOLDS *** \n")
        metrics = self.evaluator.compute_metrics()
        print("\n Mean ............ : {:.4f}".format(metrics["mean"]))
        print(" Median .......... : {:.4f}".format(metrics["median"]))
        print(" Trimean ......... : {:.4f}".format(metrics["trimean"]))
        print(" Best 25% ........ : {:.4f}".format(metrics["bst25"]))
        print(" Worst 25% ....... : {:.4f}".format(metrics["wst25"]))
        print(" Percentile 95 ... : {:.4f} \n".format(metrics["wst5"]))





if __name__ == '__main__':
    trainer = Tester()
    trainer.eval()

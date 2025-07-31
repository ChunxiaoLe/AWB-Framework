import yaml
import torch
from torch.utils.data import DataLoader
"定义模型"
from registry import build_model, build_dataloaders
"定义loss"
from loss.Loss import LossIllum
from Logger import Logger
from EarlyStopping import EarlyStopping
from config import parse_args
from model.MobileIE import MobileIENet




class Trainer:
    def __init__(self):
        self.args = parse_args()
        self.device = self.args.device
        self.logger = Logger(self.args)

        # 构建模型
        model_name = self.args.config['model']['name']
        model_cfg = self.args.config['model'].get(model_name, {})
        self.model = build_model(model_name, **model_cfg).to(self.device)
        if self.args.pretrained:
            self.model.load_state_dict(torch.load(self.args.pretrained), strict=False)

        # 损失、优化器、学习率调度
        self.loss = LossIllum()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=self.args.lr_period, T_mult=self.args.lr_mult, eta_min=self.args.min_lr)

        # 数据加载
        dataset_name = self.args.config['dataset']['name']
        dataset_config = self.args.config['dataset'][dataset_name]
        self.train_loader, self.val_loader = build_dataloaders(dataset_name=dataset_name, dataset_config=dataset_config, opt=self.args)


        # 早停
        self.early_stopping = EarlyStopping(patience=self.args.patience, min_delta=self.args.min_delta,restore_best_weights=True)

    def train_epoch(self):
        self.model.train()
        losses = []
        for data in self.train_loader:
            img, img_gt, label, _ = data
            img, label = img.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(img)
            loss = self.loss(out, label)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return sum(losses) / len(losses)

    def validate(self):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for data in self.val_loader:
                img, img_gt, label, _ = data
                img, label = img.to(self.device), label.to(self.device)
                out = self.model(img)
                loss = self.loss(out, label)
                losses.append(loss.item())
        return sum(losses) / len(losses)

    def run(self):
        best_loss = float('inf')
        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.logger.info(f"Epoch {epoch+1}/{self.args.epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), f"{self.args.save_dir}/best_model.pth")
                self.logger.info("Saved best model.")

            self.scheduler.step()

            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stop at epoch {epoch+1}")
                break




if __name__ == '__main__':

    trainer = Trainer()
    trainer.run()

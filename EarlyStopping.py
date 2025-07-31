class EarlyStopping:
    """早停机制类"""

    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
            if self.verbose:
                print(f'EarlyStopping: 验证损失改善到 {val_loss:.6f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: 验证损失未改善 {self.counter}/{self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print('EarlyStopping: 已恢复最佳权重')

    def save_checkpoint(self, model):
        """保存最佳模型权重"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()

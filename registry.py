from torch.utils.data import DataLoader

""
"------------------------------注册模型------------------------------"
MODEL_REGISTRY = {}

def register_model(name=None):
    def decorator(cls):
        model_name = name if name else cls.__name__
        if model_name in MODEL_REGISTRY:
            raise KeyError(f"Model {model_name} already registered")
        MODEL_REGISTRY[model_name] = cls
        return cls
    return decorator


def build_model(name, *args, **kwargs):
    print("DEBUG: 当前注册的模型 variants:", list(MODEL_REGISTRY.keys()))
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    cls = MODEL_REGISTRY[name]
    return cls(*args, **kwargs)


"------------------------------注册数据集------------------------------"
def create_dataset(dataset_name, **kwargs):
    module = __import__(f"dataset.{dataset_name}Dataset", fromlist=[f"{dataset_name}Dataset"])
    DatasetClass = getattr(module, f"{dataset_name}Dataset")
    return DatasetClass(**kwargs)



def build_dataloaders(dataset_name, dataset_config, opt):
    # 创建训练数据集
    train_dataset = create_dataset(dataset_name,mode='train', **dataset_config)

    # 创建测试数据集
    test_dataset = create_dataset(dataset_name, mode='test', **dataset_config)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset,batch_size=opt.train_bz,shuffle=True,num_workers=opt.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=opt.test_bz,shuffle=False, num_workers=opt.num_workers)

    return train_loader, test_loader

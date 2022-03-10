import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from test_model import *
from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap
from pl_bolts.datasets import DummyDataset


class MyDummyData(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def train_dataloader(self):
        ds = DummyDataset((8, 256, 1024), (1, 256, 1024))
        dl = torch.utils.data.DataLoader(ds, batch_size=2)
        return dl

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = PMFNet()

    def forward(self,x):
        return self.model(x)

    def training_step(self, batch, batch_index):
        x, y = batch
        print(x.shape, y.shape)
        res= self.forward(x)


    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters())

if __name__ == '__main__':
    dummy_dataset = MyDummyData()
    model = MyModel()

    trainer = pl.Trainer(gpus=[4, 5],
                         strategy='ddp',
                         precision=16)
    trainer.fit(model, dummy_dataset)
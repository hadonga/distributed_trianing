# This is a distributed training test using pl dummy dataset
# Power volume is critical for stable training.

import pytorch_lightning as pl
from test_model import *
from pl_bolts.datasets import DummyDataset

batch_size= 1

class MyDummyData(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def train_dataloader(self):
        ds = DummyDataset((8, 256, 1024))
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)
        return dl

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = PMFNet()

    def forward(self,x,y):
        return self.model(x,y)

    def training_step(self, batch, batch_index):
        x= batch[0]
        y= torch.ones((batch_size,256,1024)).long().cuda()
        img, pcd = x[:,:3,:,:], x[:,3:,:,:]
        # print(img.shape, pcd.shape, y.shape)
        res= self.forward(img, pcd)
        loss = F.cross_entropy(res, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters())

if __name__ == '__main__':
    dummy_dataset = MyDummyData()
    model = MyModel()

    trainer = pl.Trainer(gpus=[3,4,5,6,7],
                         strategy='ddp',
                         precision=16)
    trainer.fit(model, dummy_dataset)
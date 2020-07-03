from dataset import F0TrackingDataset, ToTensor
from trainer import F0TrackingModel
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers


dataset = F0TrackingDataset(len_song=10, transform=ToTensor())
tb_logger = pl_loggers.TensorBoardLogger('./logs/')
trainer = pl.trainer.Trainer(logger=tb_logger)
model = F0TrackingModel(batch_size=64, dataset=dataset)
trainer.fit(model)


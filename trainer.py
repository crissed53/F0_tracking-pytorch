import pytorch_lightning as pl
from model.model import SalomonF0Tracker
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset import F0TrackingDataset, ToTensor
import numpy as np
import torch
import os
from loss import CrossEntropyLossCustom


class F0TrackingModel(pl.LightningModule):
    """
    class LitModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def train_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True,
                                transform=transforms.ToTensor()), batch_size=32)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
    """

    def __init__(self, batch_size: int, dataset: F0TrackingDataset,
                 model_path: str = './save/models', val_split_ratio: float = 0.2):
        super().__init__()
        self.net: SalomonF0Tracker = SalomonF0Tracker()

        self.dataset = dataset
        self.batch_size = batch_size
        self.model_path = model_path
        os.makedirs(self.model_path, exist_ok=True)

        self.loss = CrossEntropyLossCustom()

        self.sampler = self._get_rand_split_sampler(val_split_ratio)

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def forward(self, x):
        return self.net.forward(x)

    def _get_rand_split_sampler(self, val_split_ratio: float, seed: int = 1):
        len_dataset = len(self.dataset)
        indices = list(range(len_dataset))
        len_val = int(val_split_ratio * len_dataset)

        np.random.seed(seed)
        np.random.shuffle(indices)

        val_indices, train_indices = indices[:len_val], indices[len_val:]

        val_sampler = SubsetRandomSampler(val_indices)
        train_sampler = SubsetRandomSampler(train_indices)

        return {'val': val_sampler, 'train': train_sampler}

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size,
                          sampler=self.sampler['train'], num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size,
                          sampler=self.sampler['val'], num_workers=12)

    def _calc_acc(self, f0: torch.Tensor, f0_hat: torch.Tensor):
        f0_hat_quantized = (f0_hat > 0.5).int()
        eq_arr = (f0 == f0_hat_quantized).int()
        num_correct = eq_arr.sum().item()
        len_arr = eq_arr.reshape(-1).shape[0]
        return num_correct / len_arr

    def _step(self, batch, batch_idx):
        hcqt = batch['hcqt']
        f0 = batch['f0']
        f0_hat = self.net(hcqt.float())
        loss = self.loss(f0_hat, f0)
        acc = self._calc_acc(f0, f0_hat)

        log = {'loss': loss, 'acc': acc}
        return {'loss': loss, 'log': log}

    def _save_model(self):
        # saves model for current epoch
        model_name = f'model__{self.current_epoch:03d}.pt'
        model_fpath = os.path.join(self.model_path, model_name)
        print(f'Saving model at: {model_fpath}')
        torch.save(self.net.state_dict(),
                   f=os.path.join(self.model_path, model_name))

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        self._save_model()

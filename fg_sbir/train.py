import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fg_sbir.dataset import SketchyFGDataset, default_transform
from fg_sbir.model import FineGrainedModel
from fg_sbir.options import opts


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    transform = default_transform(opts.max_size)

    train_dataset = SketchyFGDataset(opts, transform=transform, split='train', view='train')
    val_query_dataset = SketchyFGDataset(opts, transform=transform, split='val', view='query')
    val_gallery_dataset = SketchyFGDataset(opts, transform=transform, split='val', view='gallery')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.workers,
        pin_memory=True,
        persistent_workers=opts.workers > 0)
    val_query_loader = DataLoader(
        dataset=val_query_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.workers,
        pin_memory=True,
        persistent_workers=opts.workers > 0)
    val_gallery_loader = DataLoader(
        dataset=val_gallery_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.workers,
        pin_memory=True,
        persistent_workers=opts.workers > 0)

    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)
    checkpoint_callback = ModelCheckpoint(
        monitor='acc_top1',
        dirpath=f'saved_models/{opts.exp_name}',
        filename='{epoch:02d}-{acc_top1:.4f}',
        mode='max',
        save_last=True)

    ckpt_path = os.path.join('saved_models', opts.exp_name, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print(f'resuming training from {ckpt_path}')

    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        min_epochs=1,
        max_epochs=opts.epochs,
        benchmark=True,
        logger=logger,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
    )

    model = FineGrainedModel(opts, train_dataset.categories)
    print('beginning fine-grained training...')
    trainer.fit(model, train_loader, [val_query_loader, val_gallery_loader], ckpt_path=ckpt_path)

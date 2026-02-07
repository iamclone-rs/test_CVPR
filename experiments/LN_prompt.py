import os
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model_LN_prompt import Model
from src.dataset_retrieval import Sketchy
from experiments.options import opts

if __name__ == '__main__':
    dataset_transforms = Sketchy.data_transform(opts)

    train_dataset = Sketchy(opts, dataset_transforms, mode='train', return_orig=False)
    val_dataset_query = Sketchy(opts, dataset_transforms, mode='val', used_cat=train_dataset.all_categories, return_orig=False, image_type='triplet')
    val_dataset_gallery = Sketchy(opts, dataset_transforms, mode='val', used_cat=train_dataset.all_categories, return_orig=False, image_type='gallery')

    train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    val_loader_query = DataLoader(dataset=val_dataset_query, batch_size=opts.batch_size, num_workers=opts.workers)
    val_loader_gallery = DataLoader(dataset=val_dataset_gallery, batch_size=opts.batch_size, num_workers=opts.workers)
    
    val_loader = [val_loader_query, val_loader_gallery]

    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='saved_models/%s'%opts.exp_name,
        filename="{epoch:02d}-{top10:.2f}",
        mode='min',
        save_last=True)

    ckpt_path = os.path.join('saved_models', opts.exp_name, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    trainer = Trainer(accelerator='gpu', devices=-1,
        min_epochs=1, max_epochs=2000,
        benchmark=True,
        logger=logger,
        # val_check_interval=10, 
        # accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback]
    )

    if ckpt_path is None:
        model = Model()
    else:
        print ('resuming training from %s'%ckpt_path)
        model = Model()

    print ('beginning training...good luck...')
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

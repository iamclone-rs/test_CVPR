import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import retrieval_average_precision
import pytorch_lightning as pl

from src.clip import clip
from experiments.options import opts

def freeze_model(m):
    m.requires_grad_(False)

def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.opts = opts
        self.clip, _ = clip.load('ViT-B/32', device=self.device)
        self.clip.apply(freeze_all_but_bn)

        # Prompt Engineering
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=0.2)

        self.best_metric = -1e3

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.clip.parameters(), 'lr': self.opts.clip_LN_lr},
            {'params': [self.sk_prompt] + [self.img_prompt], 'lr': self.opts.prompt_lr}])
        return optimizer

    def forward(self, data, dtype='image'):
        if dtype == 'image':
            feat = self.clip.encode_image(
                data, self.img_prompt.expand(data.shape[0], -1, -1))
        else:
            feat = self.clip.encode_image(
                data, self.sk_prompt.expand(data.shape[0], -1, -1))
        return feat

    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        
        if dataloader_idx == 0:
            img_feat = self.forward(img_tensor, dtype='image')
            sk_feat = self.forward(sk_tensor, dtype='sketch')
            neg_feat = self.forward(neg_tensor, dtype='image')

            loss = self.loss_fn(sk_feat, img_feat, neg_feat)
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            return {'type': 'query', 'sk_feat': sk_feat, 'category': category}
        else:
            img_feat = self.forward(img_tensor, dtype='image')
            return {'type': 'gallery', 'img_feat': img_feat, 'category': category}

    def validation_epoch_end(self, val_step_outputs):
        # PTL returns list of lists when multiple val_dataloaders are used
        if not isinstance(val_step_outputs, list) or len(val_step_outputs) < 2:
            print("Warning: Expected multiple val dataloaders for query/gallery split")
            return
            
        queries_out = val_step_outputs[0]
        gallery_out = val_step_outputs[1]

        if len(queries_out) == 0 or len(gallery_out) == 0:
            return

        query_feat_all = torch.cat([x['sk_feat'] for x in queries_out])
        query_cat_all = np.array(sum([list(x['category']) for x in queries_out], []))
        
        gallery_feat_all = torch.cat([x['img_feat'] for x in gallery_out])
        gallery_cat_all = np.array(sum([list(x['category']) for x in gallery_out], []))


        ## mAP category-level SBIR Metrics
        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all))
        p_100 = torch.zeros(len(query_feat_all))

        for idx, sk_feat in enumerate(query_feat_all):
            category = query_cat_all[idx]
            distance = -1*self.distance_fn(sk_feat.unsqueeze(0), gallery)
            target = torch.zeros(len(gallery), dtype=torch.bool)
            target[np.where(gallery_cat_all == category)] = True
            
            # mAP@all
            ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu())

            # P@100
            # distance holds scores, higher is better
            # we need top 100
            sorted_idx = torch.argsort(distance, descending=True)[:100]
            # count how many relevant items in top 100
            # Ensure indices are on CPU to match target (CPU)
            relevant_count = torch.sum(target[sorted_idx.cpu()])
            p_100[idx] = relevant_count / 100.0
        
        mAP = torch.mean(ap)
        mean_p_100 = torch.mean(p_100)
        
        # Calculate max possible P@100 for debugging
        # Count how many relevant items exist for each query in the current gallery
        relevant_totals = torch.zeros(len(query_feat_all))
        for idx, category in enumerate(query_cat_all):
             relevant_totals[idx] = np.sum(gallery_cat_all == category)
        
        avg_relevant = torch.mean(relevant_totals)
        avg_max_p100 = torch.mean(torch.clamp(relevant_totals, max=100) / 100.0)

        self.log('mAP', mAP, prog_bar=True)
        self.log('P@100', mean_p_100, prog_bar=True)

        if self.global_step > 0:
            self.best_metric = self.best_metric if  (self.best_metric > mAP.item()) else mAP.item()
        
        print(f'\nStats - Avg relevant items/class in val batch: {avg_relevant:.1f}. Max possible P@100: {avg_max_p100:.4f}')
        print ('\nmAP@all: {:.4f}, P@100: {:.4f}, Best mAP: {:.4f}'.format(mAP.item(), mean_p_100.item(), self.best_metric))

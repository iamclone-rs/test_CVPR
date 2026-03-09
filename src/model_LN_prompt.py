import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

try:
    from torchmetrics.retrieval import retrieval_average_precision
except ImportError:
    from torchmetrics.functional import retrieval_average_precision

from src.clip import clip
from experiments.options import opts


def enable_layer_norm_training(module):
    module.requires_grad_(False)
    for layer in module.modules():
        if isinstance(layer, nn.LayerNorm):
            if layer.weight is not None:
                layer.weight.requires_grad_(True)
            if layer.bias is not None:
                layer.bias.requires_grad_(True)


class Model(pl.LightningModule):
    def __init__(self, train_categories):
        super().__init__()

        self.opts = opts
        self.train_categories = list(train_categories)
        self.category_to_idx = {category: idx for idx, category in enumerate(self.train_categories)}
        self.map_metric_name, self.map_top_k, self.precision_metric_name, self.precision_top_k = self._resolve_eval_config()

        self.clip, _ = clip.load('ViT-B/32', device=self.device)
        self.sketch_encoder = copy.deepcopy(self.clip.visual)

        self.clip.requires_grad_(False)
        self.sketch_encoder.requires_grad_(False)
        enable_layer_norm_training(self.clip.visual)
        enable_layer_norm_training(self.sketch_encoder)

        # Category-level ZS-SBIR learns one prompt per modality.
        self.sk_prompt = nn.Parameter(0.02 * torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(0.02 * torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.cosine_distance, margin=self.opts.margin)

        class_prompts = [f"a photo of a {self._format_category_name(category)}" for category in self.train_categories]
        self.register_buffer('class_text_tokens', clip.tokenize(class_prompts), persistent=False)

        self.best_metric = -1e3
        self._cached_text_features = None
        self._reset_validation_buffers()

    def _resolve_eval_config(self):
        dataset_name = self.opts.dataset.lower()
        if dataset_name == 'sketchy':
            return 'mAP@200', 200, 'P@200', 200
        if dataset_name == 'tuberlin':
            return 'mAP@all', None, 'P@100', 100
        return 'mAP@all', None, 'P@200', 200

    def _format_category_name(self, category):
        return category.replace('_', ' ').replace('-', ' ')

    def _reset_validation_buffers(self):
        self.val_query_features = []
        self.val_query_categories = []
        self.val_gallery_features = []
        self.val_gallery_categories = []

    def _category_targets(self, categories):
        return torch.tensor(
            [self.category_to_idx[category] for category in categories],
            device=self.device,
            dtype=torch.long)

    def _encode_visual(self, encoder, data, prompt):
        encoder_dtype = encoder.conv1.weight.dtype
        prompt_batch = prompt.unsqueeze(0).expand(data.shape[0], -1, -1)
        return encoder(data.type(encoder_dtype), prompt_batch.type(encoder_dtype))

    def _get_text_features(self):
        if (
            self._cached_text_features is None or
            self._cached_text_features.device != self.class_text_tokens.device
        ):
            with torch.no_grad():
                text_features = self.clip.encode_text(self.class_text_tokens)
                self._cached_text_features = F.normalize(text_features, dim=-1)
        return self._cached_text_features

    def cosine_distance(self, x, y):
        return 1.0 - F.cosine_similarity(x, y)

    def retrieval_score(self, query_feat, gallery_feat):
        # Keep scores positive because torchmetrics masks target entries when preds <= 0.
        return 1.0 + F.cosine_similarity(query_feat, gallery_feat)

    def classification_loss(self, features, labels):
        image_features = F.normalize(features, dim=-1)
        text_features = self._get_text_features()
        logits = self.clip.logit_scale.exp() * image_features @ text_features.t()
        return F.cross_entropy(logits, labels)

    def configure_optimizers(self):
        layer_norm_params = [
            param for param in list(self.clip.visual.parameters()) + list(self.sketch_encoder.parameters())
            if param.requires_grad
        ]
        optimizer = torch.optim.Adam([
            {'params': layer_norm_params, 'lr': self.opts.clip_LN_lr},
            {'params': [self.sk_prompt, self.img_prompt], 'lr': self.opts.prompt_lr},
        ])
        return optimizer

    def forward(self, data, dtype='image'):
        if dtype == 'image':
            return self._encode_visual(self.clip.visual, data, self.img_prompt)
        return self._encode_visual(self.sketch_encoder, data, self.sk_prompt)

    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        labels = self._category_targets(category)

        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        triplet_loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        img_cls_loss = self.classification_loss(img_feat, labels)
        sk_cls_loss = self.classification_loss(sk_feat, labels)
        loss = triplet_loss + self.opts.lambda_cls * (img_cls_loss + sk_cls_loss)

        self.log('train_loss', loss, batch_size=sk_tensor.size(0))
        self.log('train_triplet_loss', triplet_loss, batch_size=sk_tensor.size(0))
        self.log('train_img_cls_loss', img_cls_loss, batch_size=sk_tensor.size(0))
        self.log('train_sk_cls_loss', sk_cls_loss, batch_size=sk_tensor.size(0))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]

        if dataloader_idx == 0:
            img_feat = self.forward(img_tensor, dtype='image')
            sk_feat = self.forward(sk_tensor, dtype='sketch')
            neg_feat = self.forward(neg_tensor, dtype='image')

            triplet_loss = self.loss_fn(sk_feat, img_feat, neg_feat)
            self.log('val_triplet_loss', triplet_loss, on_step=False, on_epoch=True, batch_size=sk_tensor.size(0))
            self.val_query_features.append(sk_feat.detach().cpu())
            self.val_query_categories.extend(category)
        else:
            img_feat = self.forward(img_tensor, dtype='image')
            self.val_gallery_features.append(img_feat.detach().cpu())
            self.val_gallery_categories.extend(category)

    def on_fit_start(self):
        self._cached_text_features = None

    def on_validation_epoch_start(self):
        self._reset_validation_buffers()

    def on_validation_epoch_end(self):
        if not self.val_query_features or not self.val_gallery_features:
            print("Warning: Expected multiple val dataloaders for query/gallery split")
            return

        query_feat_all = torch.cat(self.val_query_features)
        query_cat_all = np.array(self.val_query_categories)

        gallery_feat_all = torch.cat(self.val_gallery_features)
        gallery_cat_all = np.array(self.val_gallery_categories)

        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all))
        precision_at_k = torch.zeros(len(query_feat_all))

        for idx, sk_feat in enumerate(query_feat_all):
            category = query_cat_all[idx]
            scores = self.retrieval_score(sk_feat.unsqueeze(0), gallery)
            target = torch.zeros(len(gallery), dtype=torch.bool)
            target[np.where(gallery_cat_all == category)] = True

            ap[idx] = retrieval_average_precision(scores.cpu(), target.cpu(), top_k=self.map_top_k)

            top_k = min(self.precision_top_k, len(gallery))
            sorted_idx = torch.argsort(scores, descending=True)[:top_k]
            relevant_count = torch.sum(target[sorted_idx.cpu()])
            precision_at_k[idx] = relevant_count / float(top_k)

        mAP = torch.mean(ap)
        mean_precision = torch.mean(precision_at_k)

        self.log('mAP', mAP, prog_bar=False)
        self.log(self.map_metric_name, mAP, prog_bar=False)
        self.log(self.precision_metric_name, mean_precision, prog_bar=False)

        self.best_metric = max(self.best_metric, mAP.item())
        print(
            f"Epoch {self.current_epoch}: {self.map_metric_name}={mAP.item():.4f}, "
            f"{self.precision_metric_name}={mean_precision.item():.4f}, Best mAP={self.best_metric:.4f}"
        )
        self._reset_validation_buffers()

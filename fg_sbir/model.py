import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.clip import clip


def enable_layer_norm_training(module):
    module.requires_grad_(False)
    for layer in module.modules():
        if isinstance(layer, nn.LayerNorm):
            if layer.weight is not None:
                layer.weight.requires_grad_(True)
            if layer.bias is not None:
                layer.bias.requires_grad_(True)


class FineGrainedModel(pl.LightningModule):
    def __init__(self, opts, train_categories):
        super().__init__()
        self.opts = opts
        self.train_categories = list(train_categories)
        self.category_to_idx = {category: idx for idx, category in enumerate(self.train_categories)}

        self.clip, _ = clip.load('ViT-B/32', device='cpu')
        self.clip.requires_grad_(False)
        enable_layer_norm_training(self.clip.visual)

        self.prompt = nn.Parameter(0.02 * torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.hard_triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=self.cosine_distance,
            margin=self.opts.margin)
        self.patch_triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=self.cosine_distance,
            margin=self.opts.patch_shuffle_margin)

        class_prompts = [f"a photo of a {self._format_category_name(category)}" for category in self.train_categories]
        self.register_buffer('class_text_tokens', clip.tokenize(class_prompts), persistent=False)

        self.best_top1 = -1.0
        self._cached_text_features = None
        self._reset_validation_buffers()

    def _format_category_name(self, category):
        return category.replace('_', ' ').replace('-', ' ')

    def _reset_validation_buffers(self):
        self.val_query_features = []
        self.val_query_categories = []
        self.val_query_instances = []
        self.val_gallery_features = []
        self.val_gallery_categories = []
        self.val_gallery_instances = []

    def _category_targets(self, categories):
        return torch.tensor(
            [self.category_to_idx[category] for category in categories],
            device=self.device,
            dtype=torch.long)

    def _get_text_features(self):
        if (
            self._cached_text_features is None or
            self._cached_text_features.device != self.class_text_tokens.device
        ):
            with torch.no_grad():
                text_features = self.clip.encode_text(self.class_text_tokens)
                self._cached_text_features = F.normalize(text_features, dim=-1)
        return self._cached_text_features

    def _encode_visual(self, images):
        encoder_dtype = self.clip.visual.conv1.weight.dtype
        prompt_batch = self.prompt.unsqueeze(0).expand(images.shape[0], -1, -1)
        return self.clip.visual(images.type(encoder_dtype), prompt_batch.type(encoder_dtype))

    def cosine_distance(self, x, y):
        return 1.0 - F.cosine_similarity(x, y)

    def retrieval_score(self, query_feat, gallery_feat):
        return 1.0 + F.cosine_similarity(query_feat, gallery_feat)

    def classification_loss(self, features, labels):
        image_features = F.normalize(features, dim=-1)
        text_features = self._get_text_features()
        logits = self.clip.logit_scale.exp() * image_features @ text_features.t()
        return F.cross_entropy(logits, labels)

    def configure_optimizers(self):
        layer_norm_params = [parameter for parameter in self.clip.visual.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam([
            {'params': layer_norm_params, 'lr': self.opts.clip_LN_lr},
            {'params': [self.prompt], 'lr': self.opts.prompt_lr},
        ])
        return optimizer

    def forward(self, images):
        return self._encode_visual(images)

    def _sample_permutations(self, batch_size, device):
        num_patches = self.opts.patch_grid * self.opts.patch_grid
        gamma1 = torch.stack([torch.randperm(num_patches, device=device) for _ in range(batch_size)], dim=0)
        gamma2 = torch.stack([torch.randperm(num_patches, device=device) for _ in range(batch_size)], dim=0)
        identical = (gamma1 == gamma2).all(dim=1)
        while identical.any():
            for idx in identical.nonzero(as_tuple=False).flatten():
                gamma2[idx] = torch.randperm(num_patches, device=device)
            identical = (gamma1 == gamma2).all(dim=1)
        return gamma1, gamma2

    def _apply_patch_permutation(self, images, permutations):
        batch_size, channels, height, width = images.shape
        grid = self.opts.patch_grid
        patch_height = height // grid
        patch_width = width // grid

        patches = images.reshape(batch_size, channels, grid, patch_height, grid, patch_width)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch_size, grid * grid, channels, patch_height, patch_width)

        gather_index = permutations.view(batch_size, grid * grid, 1, 1, 1).expand_as(patches)
        shuffled = patches.gather(1, gather_index)
        shuffled = shuffled.reshape(batch_size, grid, grid, channels, patch_height, patch_width)
        shuffled = shuffled.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, channels, height, width)
        return shuffled

    def patch_shuffling_loss(self, sketch_images, photo_images):
        gamma1, gamma2 = self._sample_permutations(sketch_images.size(0), sketch_images.device)
        shuffled_sketch = self._apply_patch_permutation(sketch_images, gamma1)
        shuffled_photo_pos = self._apply_patch_permutation(photo_images, gamma1)
        shuffled_photo_neg = self._apply_patch_permutation(photo_images, gamma2)

        sketch_features = self.forward(shuffled_sketch)
        positive_features = self.forward(shuffled_photo_pos)
        negative_features = self.forward(shuffled_photo_neg)
        return self.patch_triplet_loss(sketch_features, positive_features, negative_features)

    def training_step(self, batch, batch_idx):
        sketch_images, positive_images, negative_images, categories, _ = batch
        labels = self._category_targets(categories)

        sketch_features = self.forward(sketch_images)
        positive_features = self.forward(positive_images)
        negative_features = self.forward(negative_images)

        hard_triplet = self.hard_triplet_loss(sketch_features, positive_features, negative_features)
        sketch_cls = self.classification_loss(sketch_features, labels)
        photo_cls = self.classification_loss(positive_features, labels)
        patch_loss = self.patch_shuffling_loss(sketch_images, positive_images)

        loss = hard_triplet + self.opts.lambda_cls * (sketch_cls + photo_cls) + self.opts.lambda_patch * patch_loss

        batch_size = sketch_images.size(0)
        self.log('train_loss', loss, batch_size=batch_size)
        self.log('train_hard_triplet', hard_triplet, batch_size=batch_size)
        self.log('train_sketch_cls', sketch_cls, batch_size=batch_size)
        self.log('train_photo_cls', photo_cls, batch_size=batch_size)
        self.log('train_patch_shuffle', patch_loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            sketch_images, categories, instance_ids = batch
            features = self.forward(sketch_images)
            self.val_query_features.append(features.detach().cpu())
            self.val_query_categories.extend(categories)
            self.val_query_instances.extend(instance_ids)
        else:
            photo_images, categories, instance_ids = batch
            features = self.forward(photo_images)
            self.val_gallery_features.append(features.detach().cpu())
            self.val_gallery_categories.extend(categories)
            self.val_gallery_instances.extend(instance_ids)

    def on_fit_start(self):
        self._cached_text_features = None

    def on_validation_epoch_start(self):
        self._reset_validation_buffers()

    def on_validation_epoch_end(self):
        if not self.val_query_features or not self.val_gallery_features:
            print('Warning: expected query and gallery validation loaders')
            return

        query_features = torch.cat(self.val_query_features)
        gallery_features = torch.cat(self.val_gallery_features)
        gallery_categories = list(self.val_gallery_categories)
        gallery_instances = list(self.val_gallery_instances)

        top1_hits = 0.0
        top5_hits = 0.0
        num_queries = len(self.val_query_instances)

        for idx, query_feature in enumerate(query_features):
            category = self.val_query_categories[idx]
            instance_id = self.val_query_instances[idx]

            category_indices = [i for i, gallery_category in enumerate(gallery_categories) if gallery_category == category]
            if not category_indices:
                continue

            category_gallery = gallery_features[category_indices]
            scores = self.retrieval_score(query_feature.unsqueeze(0), category_gallery)
            sorted_local_indices = torch.argsort(scores, descending=True)
            ranked_global_indices = [category_indices[i.item()] for i in sorted_local_indices]
            ranked_instances = [gallery_instances[i] for i in ranked_global_indices]

            if ranked_instances and ranked_instances[0] == instance_id:
                top1_hits += 1.0
            if instance_id in ranked_instances[:5]:
                top5_hits += 1.0

        acc_top1 = top1_hits / max(num_queries, 1)
        acc_top5 = top5_hits / max(num_queries, 1)
        acc_top1_tensor = torch.tensor(acc_top1, device=self.device)
        acc_top5_tensor = torch.tensor(acc_top5, device=self.device)

        self.log('acc_top1', acc_top1_tensor, prog_bar=False)
        self.log('acc_top5', acc_top5_tensor, prog_bar=False)

        self.best_top1 = max(self.best_top1, acc_top1)
        print(
            f"Epoch {self.current_epoch}: Top-1={acc_top1 * 100.0:.2f}, "
            f"Top-5={acc_top5 * 100.0:.2f}, Best Top-1={self.best_top1 * 100.0:.2f}"
        )
        self._reset_validation_buffers()

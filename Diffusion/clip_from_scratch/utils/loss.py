import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.prev_num_logits = 0
        self.labels = None

    @staticmethod
    def _get_logits(image_features, text_features, logit_scale, logit_bias=None):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        if logit_bias is not None:
            logits_per_image += logit_bias
            logits_per_text += logit_bias

        return logits_per_image, logits_per_text

    def get_labels(self, device, num_logits):
        if self.prev_num_logits != num_logits:
            self.labels = torch.arange(num_logits, device=device, dtype=torch.long)
            self.prev_num_logits = num_logits

        return self.labels

    def forward(self, image_features, text_features, logit_scale, logit_bias=None, output_dict=False):
        logits_per_image, logits_per_text = self._get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_labels(image_features.device, logits_per_image.shape[0])
        loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

        if output_dict:
            return {'contrastive_loss': loss}
        else:
            return loss
import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard Contrastive Loss
def contrastive_loss(text_embeds, image_embeds, temperature=0.1):
    # Normalize embeddings
    text_embeds = F.normalize(text_embeds, dim=-1)
    image_embeds = F.normalize(image_embeds, dim=-1)

    # Compute cosine similarity
    logits = torch.matmul(text_embeds, image_embeds.T) / temperature

    # Create labels (diagonal is positive pair)
    labels = torch.arange(logits.size(0)).to(logits.device)

    # Compute loss
    loss = F.cross_entropy(logits, labels)
    return loss

# Contrastive Loss with Learnable Temperature
class ContrastiveLossWithLearnableTemp(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(0.1))

    def forward(self, text_embeds, image_embeds):
        text_embeds = F.normalize(text_embeds, dim=-1)
        image_embeds = F.normalize(image_embeds, dim=-1)
        logits = torch.matmul(text_embeds, image_embeds.T) / self.temperature
        labels = torch.arange(logits.size(0)).to(logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss

# Contrastive Loss with Hard Negative Mining
def contrastive_loss_with_hard_negatives(text_embeds, image_embeds, temperature=0.1):
    text_embeds = F.normalize(text_embeds, dim=-1)
    image_embeds = F.normalize(image_embeds, dim=-1)

    logits = torch.matmul(text_embeds, image_embeds.T) / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)

    # Example: Filter top 5 hardest negatives
    hard_negatives = torch.topk(-logits, k=5, dim=-1).values

    loss = F.cross_entropy(logits, labels)
    return loss

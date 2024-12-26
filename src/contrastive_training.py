import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from contrastive_loss import contrastive_loss  # Import the function from contrastive_loss.py

# Define projection layers for text and image embeddings
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)

# Training loop
def train_contrastive_model(text_embeds, image_embeds, num_epochs=10, device='cpu'):
    text_embeds, image_embeds = text_embeds.to(device), image_embeds.to(device)
    text_projection = ProjectionHead(768, 256).to(device)
    image_projection = ProjectionHead(512, 256).to(device)
    optimizer = Adam(list(text_projection.parameters()) + list(image_projection.parameters()), lr=1e-3)

    for epoch in range(num_epochs):
        projected_texts = text_projection(text_embeds)
        projected_images = image_projection(image_embeds)
        
        # Use the contrastive_loss function imported from contrastive_loss.py
        loss = contrastive_loss(projected_texts, projected_images)  # Call the imported loss function
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    torch.save(text_projection.state_dict(), "text_projection_model.pt")
    torch.save(image_projection.state_dict(), "image_projection_model.pt")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_embeds = torch.load("text_embeddings_japanese.pt").to(device)
    image_embeds = torch.load("image_embeddings_resnet.pt").to(device)

    train_contrastive_model(text_embeds, image_embeds, num_epochs=10, device=device)

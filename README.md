# Simple Contrastive Learning System for Text and Image Embeddings

This project demonstrates contrastive learning between text and image embeddings using PyTorch. The goal is to train a model to align these embeddings through a contrastive loss function. The model learns to bring similar text and image embeddings closer while pushing dissimilar ones apart.

## Project Structure

contrastive-learning/ ├── data/ │ ├── text_embeddings.pt │ ├── image_embeddings.pt ├── src/ │ ├── init.py │ ├── preprocessing.py │ ├── contrastive_training.py │ ├── contrastive_loss.py │ └── utils.py ├── requirements.txt └── README.md

markdown
Copy code

- **`data/`**: Contains example text and image embeddings.
- **`src/`**: Contains the source code for preprocessing, training, and contrastive loss function.
- **`requirements.txt`**: Lists the required dependencies to run the project.
- **`README.md`**: Provides an overview and instructions for running the project.

## Prerequisites

Before running the project, make sure you have Python 3.7+ installed. You will also need the following dependencies:

- **PyTorch** for model training.
- **NumPy** for data processing.

## Installation

Follow these steps to set up the project:

### 1. Clone the repository

```bash
git clone https://github.com/your-username/contrastive-learning.git
cd contrastive-learning
2. Install Dependencies
Install all the required Python packages using pip:

bash
Copy code
pip install -r requirements.txt
3. Prepare the Data
You will need to generate or provide your own text and image embeddings. The preprocessing.py file provides a simple example of how to generate random embeddings.

Run the following script to generate sample embeddings:

bash
Copy code
python src/preprocessing.py
This will create two files in the data/ directory:

text_embeddings.pt: Contains the text embeddings.
image_embeddings.pt: Contains the image embeddings.
Replace these with your own embeddings if you have real data.

4. Training the Model
Once you have prepared the embeddings, you can start training the contrastive model. Run the following script to train the model:

bash
Copy code
python src/contrastive_training.py
This will load the embeddings from the data/ directory, train the model for 10 epochs, and save the trained projection models for both text and image embeddings in the models/ directory.

The training loop will print the loss for each epoch, and at the end, you will have the following saved models:

models/text_projection_model.pt: The trained projection model for text embeddings.
models/image_projection_model.pt: The trained projection model for image embeddings.
5. Evaluate the Model
You can evaluate the trained models by loading them and testing with new data. For evaluation, you can modify the contrastive_training.py script to use the saved projection models and evaluate them with new embeddings.

Project Files
preprocessing.py
This file contains functions to process raw data and generate embeddings. In this example, the embeddings are randomly generated.

python
Copy code
import torch
import numpy as np

def preprocess_text_data(text_data):
    # Example preprocessing: Convert text data to embeddings (dummy example)
    text_embeddings = torch.randn(len(text_data), 768)  # Fake embeddings for demo
    torch.save(text_embeddings, 'data/text_embeddings.pt')

def preprocess_image_data(image_data):
    # Example preprocessing: Convert image data to embeddings (dummy example)
    image_embeddings = torch.randn(len(image_data), 512)  # Fake embeddings for demo
    torch.save(image_embeddings, 'data/image_embeddings.pt')

# Example usage
if __name__ == "__main__":
    text_data = ["sample text 1", "sample text 2"]
    image_data = ["image1.jpg", "image2.jpg"]
    preprocess_text_data(text_data)
    preprocess_image_data(image_data)
contrastive_loss.py
This file defines the contrastive loss function used during training. The loss aligns the text and image embeddings by minimizing the distance between corresponding pairs.

python
Copy code
import torch
import torch.nn.functional as F

def contrastive_loss(text_embeds, image_embeds, temperature=0.1):
    text_embeds = F.normalize(text_embeds, dim=-1)
    image_embeds = F.normalize(image_embeds, dim=-1)

    logits = torch.matmul(text_embeds, image_embeds.T) / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)

    loss = F.cross_entropy(logits, labels)
    return loss
contrastive_training.py
This file contains the training loop for the contrastive learning model. It uses the contrastive_loss.py function to train the model.

python
Copy code
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from contrastive_loss import contrastive_loss
from src.utils import load_embeddings

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)

def train_contrastive_model(text_embeds, image_embeds, num_epochs=10, device='cpu'):
    text_embeds, image_embeds = text_embeds.to(device), image_embeds.to(device)
    text_projection = ProjectionHead(768, 256).to(device)
    image_projection = ProjectionHead(512, 256).to(device)
    optimizer = Adam(list(text_projection.parameters()) + list(image_projection.parameters()), lr=1e-3)

    for epoch in range(num_epochs):
        projected_texts = text_projection(text_embeds)
        projected_images = image_projection(image_embeds)
        loss = contrastive_loss(projected_texts, projected_images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    torch.save(text_projection.state_dict(), "models/text_projection_model.pt")
    torch.save(image_projection.state_dict(), "models/image_projection_model.pt")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_embeds, image_embeds = load_embeddings(device)
    train_contrastive_model(text_embeds, image_embeds, num_epochs=10, device=device)
utils.py
This file contains utility functions to load the embeddings from disk.

python
Copy code
import torch

def load_embeddings(device):
    text_embeds = torch.load("data/text_embeddings.pt").to(device)
    image_embeds = torch.load("data/image_embeddings.pt").to(device)
    return text_embeds, image_embeds
License
This project is licensed under the MIT License.

markdown
Copy code

### Explanation:

- **`preprocessing.py`**: Generates fake embeddings for text and images (you can replace this with actual embeddings).
- **`contrastive_loss.py`**: Implements the contrastive loss function that helps the model learn to align the embeddings.
- **`contrastive_training.py`**: Contains the training logic for contrastive learning.
- **`utils.py`**: A helper file to load the preprocessed data (embeddings).
- **`README.md`**: Detailed instructions on how to set up, run, and evaluate the project.

### Usage:

1. **Install dependencies**: 
   ```bash
   pip install -r requirements.txt
Prepare embeddings (or use your own):
bash
Copy code
python src/preprocessing.py
Train the model:
bash
Copy code
python src/contrastive_training.py
This setup gives you a simple system for contrastive learning between text and image embeddings using PyTorch.

from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from PIL import Image
import torch
import os

# Initialize tokenizer and text model
tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
text_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')

# Initialize image model (ResNet)
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()  # Remove classification head

# Define image preprocessing transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Function to preprocess text
def preprocess_text(text):
    """
    Tokenize and generate text embeddings using a pre-trained BERT model.
    :param text: Input text (Japanese)
    :return: Text embedding tensor
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = text_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Function to preprocess image
def preprocess_image(image_path):
    """
    Preprocess an image and extract features using a pre-trained ResNet model.
    :param image_path: Path to the image file
    :return: Image feature tensor
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    tensor = image_transform(image).unsqueeze(0)
    with torch.no_grad():
        features = resnet(tensor)
    return features

# Function to preprocess text and image pairs
def preprocess_data(texts, image_paths):
    """
    Process a batch of text and image pairs.
    :param texts: List of input texts
    :param image_paths: List of image file paths
    :return: Two tensors: text_embeddings, image_embeddings
    """
    text_embeddings = torch.cat([preprocess_text(text) for text in texts])
    image_embeddings = torch.cat([preprocess_image(image_path) for image_path in image_paths])
    return text_embeddings, image_embeddings

# Example usage
if __name__ == "__main__":
    texts = ["こんにちは", "世界"]
    image_paths = ["Outer Space III 1080.png", "Sascha 📚🌹✝️🌹📽️ on Twitter.png"]

    text_embeds, image_embeds = preprocess_data(texts, image_paths)

    print(f"Text Embeddings Shape: {text_embeds.shape}")
    print(f"Image Embeddings Shape: {image_embeds.shape}")

    torch.save(text_embeds, "text_embeddings_japanese.pt")
    torch.save(image_embeds, "image_embeddings_resnet.pt")

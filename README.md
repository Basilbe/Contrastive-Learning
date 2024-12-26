# Contrastive Learning for Text and Image Embeddings

## Overview
This project implements a simple contrastive learning framework to align text and image embeddings using PyTorch. The approach trains the model using contrastive loss to bring similar embeddings closer together while pushing dissimilar ones apart. It is designed to be modular and extensible for research and practical applications.

## Features
- **Contrastive Loss Function**: Implements a temperature-scaled contrastive loss to align embeddings effectively.
- **Projection Heads**: Includes linear and non-linear projection heads to map embeddings into a shared latent space.
- **Random Embedding Generator**: Provides utilities to generate random embeddings for demonstration and testing.
- **Lightweight and Modular**: Easy-to-understand codebase with scope for customization.

## Technologies Used
- **PyTorch**: Deep learning framework for implementing models and loss functions.
- **Python**: Core programming language for all functionalities.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- PyTorch (`pip install torch`)
- Additional libraries listed in `requirements.txt`

### Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/contrastive-learning.git
    cd contrastive-learning
    ```

2. **Install Required Packages**:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. **Generate Random Embeddings**:
    ```bash
    python src/preprocessing.py
    ```
    This will create:
    - `data/text_embeddings.pt`: Randomly generated text embeddings.
    - `data/image_embeddings.pt`: Randomly generated image embeddings.

2. **Train the Contrastive Model**:
    ```bash
    python src/contrastive_training.py
    ```
    - Embeddings are loaded from `data/`.
    - Trained projection models are saved in `models/`.

3. **Visualize Results**:
    Use utilities in the `src/utils.py` module to analyze and visualize results.

### Project Structure

```plaintext
contrastive-learning/
├── data/                    # Stores generated or preprocessed embeddings
├── models/                  # Stores trained models
├── src/                     # Core source files
│   ├── __init__.py          # Module initializer
│   ├── preprocessing.py     # Preprocessing scripts
│   ├── contrastive_training.py # Training script
│   ├── contrastive_loss.py  # Contrastive loss implementation
│   ├── utils.py             # Utility functions
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

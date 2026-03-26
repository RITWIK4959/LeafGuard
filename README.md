# LeafGuard

LeafGuard is a plant disease prediction model using Convolutional Neural Networks (CNN). This repository contains code for training, evaluating, and deploying a model that detects common plant diseases from leaf images.

## Contents
- `main.py`: Entry point for training and inference pipelines.
- `notebook.ipynb`: Exploratory analysis and model development.
- `plant_disease_cnn_model.h5`: Pretrained model weights (Keras).
- `requirements.txt`: Python dependencies.

## Setup
1. Create and activate virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install requirements
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- Train model: `python main.py --train`
- Evaluate model: `python main.py --evaluate`
- Predict from image: `python main.py --predict path/to/image.jpg`

## GitHub
Project repository: https://github.com/RITWIK4959/LeafGuard

## License
MIT

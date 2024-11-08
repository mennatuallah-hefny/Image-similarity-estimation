# Image Similarity Estimation

This project demonstrates the use of a Siamese Network for estimating image similarity. The network is structured with three identical subnetworks to process three different images: two similar images (anchor and positive samples) and one unrelated image (negative example). The goal is to train the model to identify and estimate the similarity between images.

## Dataset

The project utilizes the **Totally Looks Like** dataset, which provides pairs of images that look similar to each other, along with images that do not resemble them.

## Project Structure

- `data/` - Contains the dataset files (ensure to download and organize images here).
- `models/` - Includes the architecture and training configurations of the Siamese Network.
- `notebooks/` - Jupyter notebooks for exploratory data analysis and model training.
- `scripts/` - Python scripts for model training, evaluation, and utility functions.

## Model Architecture

The Siamese Network consists of three identical subnetworks that learn from the images:
- **Anchor**: The primary image.
- **Positive**: An image similar to the anchor.
- **Negative**: An unrelated image for comparison.

Each subnetwork is trained with shared weights to generate embeddings that represent the images in a high-dimensional space. These embeddings are then compared using a distance metric (such as Euclidean or cosine similarity) to evaluate similarity.

## Setup

### Requirements

To install the necessary libraries, run:
```bash
pip install -r requirements.txt

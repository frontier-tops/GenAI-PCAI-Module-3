#!/bin/bash

# Create /models directory if it doesn't exist
MODEL_DIR="models/"
#mkdir -p "$MODEL_DIR"

# Define URLs
URLS=(
  "https://github.com/Ribin-Baby/brats_mri_generative_diffusion_monai/releases/download/Monai-models/model.pt"
  "https://github.com/Ribin-Baby/brats_mri_generative_diffusion_monai/releases/download/Monai-models/model_autoencoder.pt"
  "https://github.com/Ribin-Baby/brats_mri_generative_diffusion_monai/releases/download/Monai-models/best_metric_model.pth"
)

# Download each file into /models
echo "Downloading files into $MODEL_DIR ..."
for url in "${URLS[@]}"; do
    echo "Downloading: $url"
    wget -P "$MODEL_DIR" "$url"
done

echo "Download completed!"

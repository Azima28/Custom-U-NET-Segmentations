```markdown
# U-Net Image Segmentation

A complete PyTorch implementation of U-Net architecture for semantic image segmentation with automatic dataset analysis and flexible mask format support.

## Features

- 🔍 **Automatic Dataset Analysis** - Detects number of classes and mask format automatically
- 🎨 **Flexible Mask Support** - Handles both RGB and grayscale masks with auto value remapping
- 🚀 **GPU Acceleration** - CUDA support with automatic CPU fallback
- 📊 **Training Visualization** - Real-time progress with loss and accuracy metrics
- 📈 **Performance Metrics** - IoU (Intersection over Union) calculation
- 🖼️ **Easy Prediction** - Simple inference pipeline with overlay visualization
- ✨ **Auto-Detection** - Automatically detects number of classes from saved models

## Architecture

The U-Net model consists of:
- **Encoder**: 4 downsampling blocks (64→128→256→512 channels)
- **Bottleneck**: 1024 channels for deep feature extraction
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: Pixel-wise classification for semantic segmentation

## Installation

```bash
pip install torch torchvision numpy matplotlib pillow opencv-python scikit-learn tqdm
```

## Dataset Structure

```
dataset/
├── images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── masks/
    ├── mask1.png
    ├── mask2.png
    └── ...
```

## Usage

### 1. Training

Configure your dataset paths in the main function:

```python
MASK_DIR = "/content/drive/MyDrive/datasets/masks"
IMAGE_DIR = "/content/drive/MyDrive/datasets/images"

# Hyperparameters
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
```

Run training:
```python
if __name__ == "__main__":
    main()
```

### 2. Check Number of Classes

Automatically detect classes from saved model:

```python
def detect_num_classes(model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    num_classes = state_dict['out.weight'].shape[0]
    return num_classes

NUM_CLASSES = detect_num_classes('/content/best_unet_model.pth')
print(f"Detected: {NUM_CLASSES} classes")
```

### 3. Prediction

Predict on new images:

```python
MODEL_PATH = '/content/drive/MyDrive/best_unet_model.pth'
IMAGE_PATH = '/content/drive/MyDrive/test_image.jpg'

prediction = predict_image(MODEL_PATH, IMAGE_PATH)
```

The function automatically detects the number of classes from the model.

## Key Features Explained

### Automatic Mask Value Remapping

Handles non-continuous mask values automatically:
- **Before**: `[0, 85, 170, 255]`
- **After**: `[0, 1, 2, 3]`

This ensures compatibility with PyTorch's CrossEntropyLoss.

### Comprehensive Training Pipeline

```
Dataset Analysis → Split Train/Val → Create DataLoaders → 
Initialize Model → Train with Validation → Save Best Model → 
Visualize Results → Calculate IoU
```

### Robust Error Handling

- CUDA error recovery with CPU fallback
- Mask validation before training starts
- Forward pass testing for model compatibility
- Memory usage monitoring

## Output Files

Training generates:
- `best_unet_model.pth` - Best model weights based on validation loss
- `training_history.png` - Loss and accuracy curves over epochs
- `predictions.png` - Sample predictions on validation set

Prediction generates:
- `prediction_result.png` - Original, mask, and overlay visualization

## Performance Metrics

- **Pixel Accuracy**: Percentage of correctly classified pixels
- **Mean IoU**: Average intersection over union across all classes
- **Class-wise Statistics**: Pixel count and percentage per class
- **Training/Validation Loss**: Cross-entropy loss tracking

## Model Architecture Details

```python
UNet(
  Encoder:
    enc1: DoubleConv(3 → 64)
    enc2: DoubleConv(64 → 128)
    enc3: DoubleConv(128 → 256)
    enc4: DoubleConv(256 → 512)
  
  Bottleneck:
    bottleneck: DoubleConv(512 → 1024)
  
  Decoder:
    dec4: DoubleConv(1024 → 512) + skip connection
    dec3: DoubleConv(512 → 256) + skip connection
    dec2: DoubleConv(256 → 128) + skip connection
    dec1: DoubleConv(128 → 64) + skip connection
  
  Output:
    out: Conv2d(64 → num_classes)
)
```

## Visualization Examples

The prediction pipeline provides three views:
1. **Original Image** - Input image
2. **Segmentation Mask** - Color-coded class predictions with colorbar
3. **Overlay** - Blended original image and prediction mask (50/50)

## Training Output Example

```
============================================================
Epoch [10/50] Summary:
  Train Loss: 0.3421 | Train Acc: 89.45%
  Val Loss:   0.4123 | Val Acc:   85.67%
  Time: 45.2s | Elapsed: 7.5m | ETA: 30.1m
  ✅ Best model saved! (Val Loss: 0.4123, Val Acc: 85.67%)
  GPU Memory: 2048.50 MB
============================================================
```

## Tips for Best Results

1. **Dataset Balance**: Ensure balanced representation of all classes
2. **Image Size**: Use 256×256 for faster training, 512×512 for better quality
3. **Batch Size**: Adjust based on GPU memory (4-16 typical range)
4. **Learning Rate**: Start with 1e-4, reduce if loss plateaus
5. **Epochs**: 50-100 epochs usually sufficient for convergence

## Common Issues & Solutions

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size or image size

### Issue: Mask values not continuous
**Solution**: Automatic remapping handles this - just verify in analysis output

### Issue: Low validation accuracy
**Solution**: Check mask-image alignment, increase epochs, adjust learning rate

## Requirements

- Python 3.7+
- PyTorch 1.7+
- CUDA (optional, for GPU acceleration)
- 4GB+ RAM (8GB+ recommended)
- GPU with 4GB+ VRAM (for faster training)

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{unet-segmentation,
  author = {Your Name},
  title = {U-Net Image Segmentation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/unet-segmentation}
}
```

## Acknowledgments

- Original U-Net paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- PyTorch implementation inspired by the deep learning community

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.

---

⭐ **Star this repo if you find it helpful!**
```
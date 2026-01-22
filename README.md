# Image Classification with Transfer Learning
PyTorch Implementation

This project implements an image classification pipeline using PyTorch with a focus on transfer learning. Two pre-trained convolutional neural network architectures are used and compared to evaluate their performance on a custom multi-class dataset.

The notebook covers data loading, model definition, training, evaluation, and visualization of results.

## Features

- Image classification using deep learning  
- Transfer learning with pre-trained models  
- Dynamic training function  
- GPU support (CUDA)  
- Accuracy visualization  
- Model comparison  

## Models Used

The following architectures are implemented:

- EfficientNet-B0  
- MobileNetV2  

Both models are fine-tuned by replacing their final classification layers to match the number of classes in the dataset.

## Methodology

### 1. Data Preparation

- Images are loaded using PyTorch `DataLoader`.  
- Dataset is split into training, validation, and test sets.  
- Images are resized and normalized.  
- Labels are assigned based on folder structure.  

### 2. Model Setup

For each model:

- Load pre-trained weights.  
- Replace the final fully connected layer.  
- Define loss function and optimizer.  

### 3. Training

A dynamic training function is used to:

- Train model for multiple epochs.  
- Validate at each epoch.  
- Track training and validation accuracy.  

### 4. Evaluation

Evaluation includes:

- Accuracy comparison between models.  
- Accuracy curves visualization.  
- Prediction visualization on test images.  

## How to Run

### Install Dependencies

```
pip install torch torchvision matplotlib scikit-learn
```

### Run Notebook

```
jupyter notebook classification.ipynb
```

## Project Structure

```
.
├── classification.ipynb
└── dataset/
    ├── class_1/
    ├── class_2/
    ├── class_3/
    └── class_4/
```

## Results

The notebook produces:

- Training vs validation accuracy plots  
- Sample predictions on test images  
- Performance comparison between EfficientNet and MobileNet  

## Possible Improvements

- Data augmentation  
- Hyperparameter tuning  
- Try deeper architectures (ResNet, DenseNet)  
- Add confusion matrix and F1-score  
- Save trained models  

## Use Case

This project is suitable for:

- Learning transfer learning in PyTorch  
- Computer vision experiments  
- Academic assignments  
- Deep learning portfolio  

## License

This project is intended for educational and research purposes.

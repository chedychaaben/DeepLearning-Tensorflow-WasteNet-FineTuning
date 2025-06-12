# WasteNet-FineTuning 🌍♻️

Welcome to **WasteNet-FineTuning**, a deep learning project for waste classification using Convolutional Neural Networks (CNNs) with TensorFlow! 🚀 This repository implements a fine-tuned ResNet50-based model to classify waste images, leveraging transfer learning and data augmentation for robust performance. Perfect for researchers and developers interested in sustainable AI solutions! 🌱

## 📖 Project Overview

This project focuses on building and fine-tuning a CNN model for binary waste classification (e.g., organic vs. recyclable). It uses a pretrained ResNet50 model on ImageNet, enhanced with data augmentation and custom layers, to process 48x48 RGB images. The model is trained and evaluated on a waste dataset, with performance visualized via confusion matrices. 📊

Key features:
- 🧠 Fine-tuned ResNet50 architecture for waste classification
- 🔄 Data augmentation (rotation, zoom, horizontal flip)
- ⚡ Optimized data loading with caching and prefetching
- ✅ Early stopping and model checkpointing for efficient training
- 📈 Confusion matrix visualization for model evaluation

## 🛠️ Setup

Follow these steps to set up the project in a Google Colab environment or locally with Python.

### Prerequisites
- Python 3.8+ 🐍
- TensorFlow 2.x
- Libraries: `numpy`, `pandas`, `seaborn`, `matplotlib`, `scikit-learn`
- Google Drive for dataset and model weight storage (if using Colab) ☁️
- Dataset: `waste_dataset` (unzipped in `/content/drive/MyDrive/waste_dataset`)

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/WasteNet-FineTuning.git
   cd WasteNet-FineTuning
   ```

2. **Install Dependencies**:
   ```bash
   pip install tensorflow numpy pandas seaborn matplotlib scikit-learn
   ```

3. **Google Colab Setup** (if applicable):
   - Mount Google Drive in Colab:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Unzip the dataset:
     ```bash
     !unzip /content/drive/MyDrive/waste_dataset.zip -d /content/drive/MyDrive/
     ```

4. **Prepare the Dataset**:
   - Ensure the `waste_dataset` folder contains `Train` and `Test` subdirectories with images.
   - Place `test_dechets.csv` in `/content/drive/MyDrive/` for evaluation.

## 🚀 Usage

1. **Run the Notebook**:
   - Open `WasteNet-FineTuning.ipynb` in Jupyter or Google Colab.
   - Execute the cells sequentially to:
     - Load and preprocess the dataset 📂
     - Build and compile the CNN model 🧠
     - Train the model with early stopping and checkpointing ⚙️
     - Evaluate performance using a confusion matrix 📊

2. **Key Steps in the Notebook**:
   - **Data Loading**: Loads RGB images (48x48) into `train_ds`, `val_ds`, and `test_ds` with a batch size of 64.
   - **Model Building**: Constructs `p_m_dechets2` using ResNet50 and custom layers (see Figure 2 in the notebook).
   - **Training**: Trains for 10 epochs with Adam optimizer, monitoring `val_accuracy`.
   - **Evaluation**: Generates predictions and visualizes the confusion matrix.

3. **Example Command** (for local execution):
   ```bash
   jupyter notebook WasteNet-FineTuning.ipynb
   ```

## 📂 Dataset

The `waste_dataset` contains:
- **Train**: Training images, split into 80% training and 20% validation.
- **Test**: Test images for evaluation.
- **test_dechets.csv**: CSV file with test data and labels for confusion matrix generation.

Ensure the dataset is structured as:
```
/content/drive/MyDrive/waste_dataset/
├── Train/
│   ├── class1/
│   ├── class2/
├── Test/
│   ├── class1/
│   ├── class2/
```

## 🧪 Multi-Input Model

The notebook also includes a multi-input model combining ResNet50 and VGG16 features for a 3-class classification task. This requires two datasets (`Dataset1` and `Dataset2`) with identical labels. Update the paths in the notebook if using this feature.

## 📈 Results

After training, the model saves the best weights to `/content/drive/MyDrive/p_m_dechets2.weights.h5`. The confusion matrix visualizes performance on the test set, with labels `O` (organic) and `R` (recyclable). Example output:
- **Accuracy**: Computed per epoch for training and validation.
- **Confusion Matrix**: Shows true vs. predicted labels.

## 🤝 Contributing

Contributions are welcome! 🙌 Feel free to:
- Open issues for bugs or feature requests 🐞
- Submit pull requests with improvements 🔧
- Share feedback on model performance or dataset quality 📬

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the awesome deep learning framework
- ImageNet for pretrained weights
- The waste classification community for inspiring sustainable AI 🌍

Happy coding and classifying! 🎉
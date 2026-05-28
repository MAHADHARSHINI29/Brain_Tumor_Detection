# 🧠 Brain Tumor Detection System

A deep learning-based web application that classifies brain MRI images to detect and identify different types of brain tumors using Convolutional Neural Networks (CNN).

## 📋 Overview

This project uses TensorFlow and Keras to build a medical image classification system that can identify four categories:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

The model is deployed as a Flask web application with an intuitive interface for uploading MRI images and receiving instant predictions.

## ✨ Features

- **Multi-class Classification**: Identifies 4 different tumor types with confidence scores
- **Real-time Prediction**: Instant analysis of uploaded MRI images
- **User-friendly Interface**: Clean, responsive web UI with image preview
- **Educational Content**: Detailed information about each tumor type, symptoms, and treatments
- **Model Performance Metrics**: Comprehensive evaluation using confusion matrices and classification reports
- **Experimental GAN Implementation**: Anomaly detection approach for medical imaging

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV, NumPy
- **Web Framework**: Flask
- **Frontend**: HTML, CSS (Bulma), JavaScript
- **Data Analysis**: Pandas, Matplotlib, Seaborn, Scikit-learn

## 📁 Project Structure

```
brain-tumor-detection/
│
├── gan.ipynb                 # Main notebook with model training
├── model_vgg_full.h5         # Trained model (architecture + weights)
├── model_vgg.json            # Model architecture
│
├── templates/
│   ├── home.html             # Landing page with upload form
│   └── classify.html         # Results page with predictions
│
├── static/
│   ├── script.js             # Frontend JavaScript
│   └── images/               # Static images
│
├── uploads/                  # Uploaded images storage
├── seg_train/                # Training dataset
└── seg_test/                 # Testing dataset
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7+
TensorFlow 2.x
Flask
OpenCV
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
Jupyter Notebook
```

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
```

2. Install required packages
```bash
pip install tensorflow flask opencv-python numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Running the Application

1. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

2. **Open `gan.ipynb`** in the Jupyter interface

3. **Run all cells** in the notebook sequentially (Cell → Run All, or run each cell one by one)
   - The notebook will load and preprocess the data
   - Train the model (or load the pre-trained model)
   - Start the Flask web server in the final cell

4. **Access the web application**
   - Once the Flask server starts, you'll see: `Running on http://127.0.0.1:5000`
   - Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

5. **Upload and classify** brain MRI images through the web interface

6. **To stop the server**: Press the stop button (■) in Jupyter Notebook or interrupt the kernel

## 📊 Model Architecture

### CNN Model
- **Input Layer**: 150x150x3 (RGB images)
- **Conv2D Layers**: 32 filters with ReLU activation
- **MaxPooling Layers**: 2x2 pooling
- **Dense Layers**: 128 neurons with ReLU
- **Output Layer**: 14 neurons with Softmax activation
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy

### Training Details
- **Epochs**: 20
- **Batch Size**: 128
- **Validation Split**: 20%
- **Image Size**: 150x150 pixels
- **Normalization**: Pixel values scaled to [0, 1]

## 📈 Model Performance

The model is evaluated using:
- Confusion Matrix
- Accuracy Score
- Precision, Recall, F1-Score
- Classification Report
- Training/Validation Loss and Accuracy Curves

## 🖼️ Dataset

The dataset consists of brain MRI images organized into four categories:
- `glioma_tumor/`
- `meningioma_tumor/`
- `no_tumor/`
- `pituitary_tumor/`

Images are preprocessed with:
- Resizing to 150x150 pixels
- BGR to RGB conversion
- Normalization (0-1 range)
- Data shuffling

## 💡 Usage

1. **Upload an MRI Image**: Click on the upload button and select a brain MRI scan
2. **Preview**: View the uploaded image before classification
3. **Classify**: Click "Detect Tumor" to get predictions
4. **Results**: View the detected tumor type, confidence score, and medical information

## 🔬 Experimental Features

### GAN Implementation
The project includes an experimental Generative Adversarial Network (GAN) for:
- Anomaly detection in medical images
- Synthetic medical image generation
- Enhanced pattern recognition

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This project is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice.

## License

This project is open source and available under the MIT License.


## Acknowledgments

- Dataset providers
- TensorFlow and Keras communities
- Medical imaging research community

---

**Note**: The entire application runs from the Jupyter notebook (`gan.ipynb`). Simply run all cells to train the model and launch the web interface.

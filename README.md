# Brain Tumor Detection & Report Generator

An AI-powered web application for detecting brain tumors from MRI images using deep learning and generating medical reports with contextual assistance from a chatbot.

## 🌐 Live Demo
Run locally using Streamlit (instructions below). The app includes:
- Tumor classification using a trained CNN model.
- Grad-CAM heatmap for visual interpretability.
- Chatbot assistant powered by OpenAI for report explanation.
- Automatically generated PDF-style medical reports.

## 📄 Model Overview
The deep learning model (Keras/TensorFlow) classifies MRI images into four categories:
- Glioma
- Meningioma
- Pituitary
- No Tumor

Input images are resized to 224x224, normalized, and passed through the CNN model. Predictions are visualized using Grad-CAM.

## ⚖️ Tech Stack
- **Frontend**: Streamlit  
- **Backend Model**: TensorFlow/Keras  
- **Image Handling**: PIL, OpenCV  
- **Explainability**: Grad-CAM  
- **Chat Assistant**: OpenAI GPT-3.5 Turbo  
- **Environment Management**: `python-dotenv`

## 📊 Features
- Patient data entry (name, ID, age, scan date).
- Upload MRI scan (.jpg/.png).
- Prediction results with class probabilities.
- Grad-CAM visualization.
- Dynamic report generation.
- Integrated medical chatbot assistant.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/brain-tumor-detector.git
cd brain-tumor-detector
```

### 2. Setup virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add environment variables
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key
```

### 5. Run the app
```bash
streamlit run app.py
```

## 📁 Project Structure
```
.
├── app.py                 # Streamlit application
├── model.ipynb            # Notebook for training and evaluation
├── classification_model.h5# Trained CNN model
├── requirements.txt       # Required Python packages
├── .env                   # Environment variables (ignored in Git)
├── .gitignore             # Git ignore file
└── dataset/               # Dataset directory (ignored in Git)
```

## 🗂️ .gitignore
Make sure the following are in your `.gitignore`:
```
.env
.venv/
dataset/
```

## 📈 Sample Output
- **Prediction**: Glioma (Confidence: 92.5%)
- **Grad-CAM**: Heatmap showing tumor region
- **Report**: AI-generated diagnostic summary

## 💪 Credits
- MRI datasets from open-access medical sources
- Model architecture inspired by standard CNN designs
- OpenAI GPT-3.5 for report explanation

## ✅ License
MIT License. See `LICENSE` file.

---

Built with ❤️ by [Your Name]

<img width="1695" height="812" alt="image" src="https://github.com/user-attachments/assets/83db3c4d-9fb9-498f-abe1-e00d111d1a12" />
🩸 Real-Time GI Bleeding Detector using Computer Vision

An AI-powered deep learning system to detect gastrointestinal (GI) bleeding from endoscopic images in real-time. This project leverages transfer learning (MobileNetV2) and a Gradio-powered interactive demo to classify images into Bleeding or Non-Bleeding, helping in faster medical diagnosis.

Project Overview
Gastrointestinal bleeding is a critical medical condition that requires rapid and accurate diagnosis. Endoscopic procedures, while effective, rely on the human eye to spot anomalies in a continuous video feed. This application leverages the power of computer vision to automate this process, providing a second, vigilant "eye" that can analyze the video stream in real time. By continuously scanning for visual markers of bleeding—such as specific color tones and textures—the system can serve as a proactive aid, potentially reducing diagnostic time and improving patient outcomes. This project serves as a proof-of-concept for a low-cost, effective, and accessible solution in medical diagnostics.

📌 Features

✅ Real-time classification of GI bleeding from endoscopic images

✅ Transfer learning using MobileNetV2 (pre-trained on ImageNet)

✅ Training + Inference pipeline in a single script

✅ User-friendly Gradio web interface for predictions

✅ Support for GPU acceleration (CUDA)

🛠 Tech Stack

Python 3.8+

PyTorch – Model training & inference

Torchvision – Pre-trained models & transforms

Gradio – Interactive web-based UI

PIL – Image processing

📂 Dataset Structure

Place your dataset in the Image/ directory:

Image/
 ├── Bleeding/
 │    ├── img1.jpg
 │    ├── img2.jpg
 │    └── ...
 └── Non-Bleeding/
      ├── img1.jpg
      ├── img2.jpg
      └── ...


Each subfolder should contain relevant images for classification.

⚙️ Installation
# Clone the repository
git clone https://github.com/ATIKWEBD/Real-Time-GI-Bleeding-Detector-using-Computer-Vision.git
cd Real-Time-GI-Bleeding-Detector-using-Computer-Vision

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)

# Install dependencies
pip install -r requirements.txt


Example requirements.txt:

torch
torchvision
gradio
Pillow

🚀 Usage

Run the app directly:

python app.py

Steps:

The script will train the model on your dataset (Image/ folder).

After training, the Gradio interface will automatically launch.

Open the provided link, upload an endoscopic image, and get predictions:

Bleeding

Non-Bleeding

📊 Model Details

Base model: MobileNetV2 (transfer learning)

Optimizer: Adam

Loss Function: CrossEntropyLoss

Epochs: 5 (configurable)

Batch Size: 32

You can tweak these parameters inside app.py.

📷 Demo (Example UI)

When launched, the app provides an interface like this:

+------------------------------------------+
| Upload Endoscopic Image [Choose File]    |
|                                          |
| [Prediction Result]                      |
|   - Bleeding: 0.92                       |
|   - Non-Bleeding: 0.08                   |
+------------------------------------------+

🤝 Contributing

Contributions are welcome! Please fork the repo and create a pull request with improvements.

📜 License

This project is licensed under the MIT License – feel free to use and modify it

Real-Time GI Bleeding Detector
This project presents a real-time application for detecting gastrointestinal (GI) bleeding from a video stream using computer vision. It is built as a diagnostic tool to assist healthcare professionals by processing visual data and alerting them to potential bleeding events. The application demonstrates a powerful and practical use of computer vision technology in the medical field.

Project Overview
Gastrointestinal bleeding is a critical medical condition that requires rapid and accurate diagnosis. Endoscopic procedures, while effective, rely on the human eye to spot anomalies in a continuous video feed. This application leverages the power of computer vision to automate this process, providing a second, vigilant "eye" that can analyze the video stream in real time. By continuously scanning for visual markers of bleeding—such as specific color tones and textures—the system can serve as a proactive aid, potentially reducing diagnostic time and improving patient outcomes. This project serves as a proof-of-concept for a low-cost, effective, and accessible solution in medical diagnostics.

Features
Real-Time Video Analysis: The application processes a live video stream frame by frame, performing instant analysis to detect signs of bleeding as they appear.

Computer Vision-Powered Detection: It uses robust computer vision algorithms to accurately identify visual indicators of GI bleeding, distinguishing them from other visual noise in the video feed.

Intuitive User Interface: The user interface, powered by Streamlit, provides a simple and clean dashboard where healthcare providers can upload or connect to a video source and monitor the detection process seamlessly.

Proof-of-Concept for Medical Aid: Designed to showcase how technology can be used as a valuable tool in a medical setting, enhancing diagnostic capabilities and supporting medical professionals.

Installation
To get a copy of the project up and running on your local machine, follow these steps.

Clone the repository:

git clone [https://github.com/ATIKWEBD/Real-Time-GI-Bleeding-Detector-using-Computer-Vision.git](https://github.com/ATIKWEBD/Real-Time-GI-Bleeding-Detector-using-Computer-Vision.git)
cd Real-Time-GI-Bleeding-Detector-using-Computer-Vision

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required libraries:
Create a requirements.txt file in your project directory with the following content:

streamlit
opencv-python
Pillow

Then, install the dependencies by running this command:

pip install -r requirements.txt

Usage
To run the application, simply execute the following command in your terminal:

streamlit run app.py

This will launch the application in your web browser, allowing you to start the real-time detection process on your chosen video stream.

License
This project is licensed under the MIT License.

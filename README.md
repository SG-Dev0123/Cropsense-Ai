🌿 CropSense AI

AI-Powered Plant Disease Diagnostics Platform using YOLOv8, Streamlit, OpenCV, and LLM-based Agricultural Chatbot.

📌 About the Project

CropSense AI is a smart agriculture web application that detects plant leaf diseases using Artificial Intelligence and Computer Vision. Users can upload a leaf image, and the system predicts the disease, confidence score, severity level, and treatment recommendations instantly.

The platform also includes an AI-powered agriculture chatbot with voice support for answering farming and crop-related questions.

🚀 Features

✅ Plant Disease Detection using YOLOv8
✅ 30 Crop Disease Classes Supported
✅ Confidence Score Prediction
✅ Severity Analysis using OpenCV
✅ Treatment Recommendations
✅ AI Agriculture Chatbot (LLM Powered)
✅ Voice Assistant Support
✅ Beautiful Streamlit UI
✅ Login & Signup Authentication
✅ Real-time Image Upload & Prediction

🧠 Technologies Used
Python
Streamlit
YOLOv8 (Ultralytics)
OpenCV
NumPy
Pillow (PIL)
Groq API
Llama 3.1 8B
HTML/CSS Styling
🌱 Supported Disease Classes
Apple Scab Leaf
Apple Rust Leaf
Bell Pepper Leaf Spot
Corn Gray Leaf Spot
Potato Early Blight
Tomato Leaf Mosaic Virus
Grape Black Rot
Squash Powdery Mildew
And many more...

Total Classes: 30

CropSense-AI/
│
├── app.py
├── data.yaml
├── requirements.txt
├── users.json
├── README.md
│
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt


🤖 AI Chatbot

The chatbot is powered by:

Groq API
Llama 3.1 8B LLM

It can answer:

Crop disease questions
Fertilizer suggestions
Fungicide recommendations
General agriculture queries
📊 Model Information
Parameter	Value
Model	YOLOv8n
Epochs	30
Classes	30
Image Size	640x640
Framework	Ultralytics
🔐 Security
Passwords stored using SHA-256 hashing
Local storage authentication
API key protection recommended using .env

👨‍💻 Developed By

Siddhi Gawade

🌾 Project Goal

To help farmers and agriculture professionals detect crop diseases quickly and accurately using AI technology.

📜 License

This project is developed for educational and research purposes.

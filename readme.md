



# AI Detection Dashboard

A web dashboard for Deepfake, Object, and Fraud Detection using Flask, SQLAlchemy, and Docker.

---

## 🚀 Quick Start

### 1. Build the Docker Image

```sh
docker build -t ai-detection-dashboard .
```

### 2. Run the Docker Container

```sh
docker run -p 5000:5000 ai-detection-dashboard
```

- The app will be available at [http://localhost:5000](http://localhost:5000)

---

## 🛠️ Features

- Deepfake detection for images and videos
- Object detection using YOLOv8
- Fraud detection for documents
- Evidence report generation
- User authentication and dashboard
- SQLite database for persistent storage

---

## 📁 Project Structure

```
ai-detection-dashboard/
│
├── app.py
├── config.py
├── database_models.py
├── deepfake_detection.py
├── object_detection.py
├── fraud_detection.py
├── evidence_report_generator.py
├── static/
│   ├── uploads/
│   └── ...
├── templates/
│   └── ...
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 👨‍💻 Author

- **Akshat Jasrotia**  [[GitHub]](https://github.com/akshatjasrotia85)
- **Priyanshu Rana**  [[GitHub]](https://github.com/priyanshurana)
- **Shivong Sharma**  [[GitHub]](https://github.com/shivongsharma)
---

Package            Version
------------------ -----------
alembic            1.17.1
blinker            1.9.0
certifi            2025.10.5
charset-normalizer 3.4.4
click              8.3.0
contourpy          1.3.3
cycler             0.12.1
filelock           3.19.1
Flask              3.0.0
Flask-Migrate      4.0.5
Flask-SQLAlchemy   3.0.5
fonttools          4.60.1
fsspec             2025.9.0
greenlet           3.2.4
hf-xet             1.2.0
huggingface-hub    0.36.0
idna               3.11
itsdangerous       2.2.0
Jinja2             3.1.6
joblib             1.5.2
kiwisolver         1.4.9
Mako               1.3.10
MarkupSafe         2.1.5
matplotlib         3.10.7
mpmath             1.3.0
networkx           3.5
numpy              1.26.4
opencv-python      4.8.1.78
packaging          25.0
pandas             2.3.3
pillow             11.3.0
pip                25.3
polars             1.35.2
polars-runtime-32  1.35.2
psutil             7.1.3
pyparsing          3.2.5
python-dateutil    2.9.0.post0
python-dotenv      1.2.1
pytz               2025.2
PyYAML             6.0.3
regex              2025.10.23
reportlab          4.4.4
requests           2.32.5
safetensors        0.6.2
scikit-learn       1.7.2
scipy              1.16.3
setuptools         80.9.0
six                1.17.0
SQLAlchemy         2.0.44
sympy              1.14.0
threadpoolctl      3.6.0
tokenizers         0.22.1
torch              2.9.0+cpu
torchaudio         2.9.0+cpu
torchcodec         0.8.1
torchvision        0.24.0+cpu
tqdm               4.67.1
transformers       4.57.1
typing_extensions  4.15.0
tzdata             2025.2
ultralytics        8.3.223
ultralytics-thop   2.0.18
urllib3            2.5.0
Werkzeug           3.0.1
wheel              0.45.1


## 📄 License

This project is licensed under the CUJ,J&K


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
### Arindam's changes:
#### To-do's
    1. making the cnn model work for deepfake detection.
        - currently used 2 models:
            one for image detection and video detection using api calls to outside.
            image model: dima806/deepfake_vs_real_image_detection (works)
            video model: Naman712/Deep-fake-detection(didnot test but should work)
    2. making the fraud detection model to work.
    3. customizing the report.
## 👨‍💻 Author

- **Akshat Jasrotia**  [[GitHub]](https://github.com/akshatjasrotia85)
- **Priyanshu Rana**  [[GitHub]](https://github.com/priyanshurana)
- **Shivong Sharma**  [[GitHub]](https://github.com/shivongsharma)


---

## 📄 License


This project is licensed under the CUJ,J&K

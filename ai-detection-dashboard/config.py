import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///detection_system.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # File upload settings
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {
        'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp',  # Images
        'mp4', 'avi', 'mov', 'mkv', 'webm',  # Videos
        'wav', 'flac', 'ogg', 'mp3', 'm4a', 'aac',  # Audio
        'pdf', 'doc', 'docx', 'json', 'csv'  # Documents
    }

    # Model paths
    YOLO_MODEL_PATH = 'yolov8n.pt'  # YOLOv8 is in root directory
    # Note: Deepfake models download automatically from Hugging Face
    FRAUD_MODEL_PATH = 'models/weights/fraud_detection.pkl'

    # Evidence report settings
    EVIDENCE_TEMPLATE_PATH = 'evidence/templates/court_evidence_template.html'
    REPORTS_FOLDER = 'evidence/exports'
    REPORT_RETENTION_DAYS = 365

    # Security
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Audit logging
    ENABLE_AUDIT_LOG = True
    LOG_RETENTION_DAYS = 730
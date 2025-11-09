from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    detection_results = db.relationship('DetectionResult', backref='user', lazy='dynamic')


class DetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    detection_type = db.Column(db.String(50), nullable=False)  # deepfake, object, fraud
    media_type = db.Column(db.String(20))  # image, video, audio - NEW FIELD
    result = db.Column(db.Text)  # JSON stored as text
    confidence = db.Column(db.Float, default=0.0)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    meta = db.Column(db.Text)  # Additional metadata as JSON

    evidence_reports = db.relationship('EvidenceReport', backref='detection', lazy='dynamic')


class EvidenceReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    detection_id = db.Column(db.Integer, db.ForeignKey('detection_result.id'), nullable=False)
    report_number = db.Column(db.String(100), unique=True, nullable=False)
    report_type = db.Column(db.String(50), default='court_evidence')  # ADD THIS LINE
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)
    file_path = db.Column(db.String(255))
    chain_of_custody = db.Column(db.Text)  # JSON
    technical_analysis = db.Column(db.Text)  # JSON
    legal_certification = db.Column(db.Text)
    report_hash = db.Column(db.String(64))  # ADD THIS LINE
    status = db.Column(db.String(20), default='pending')  # ADD THIS LINE



class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    action = db.Column(db.String(100), nullable=False)
    resource_type = db.Column(db.String(50))
    resource_id = db.Column(db.Integer)
    details = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(45))


class SystemConfig(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.Text)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
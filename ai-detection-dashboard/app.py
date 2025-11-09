from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import json

from config import Config
from database_models import db, User, DetectionResult, EvidenceReport, AuditLog
from deepfake_detection import DeepfakeDetector
from object_detection import ObjectDetector
from fraud_detection import FraudDetector
from evidence_report_generator import EvidenceReportGenerator

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

# Initialize detectors
deepfake_detector = DeepfakeDetector()
object_detector = ObjectDetector(model_path=app.config['YOLO_MODEL_PATH'])
fraud_detector = FraudDetector()
report_generator = EvidenceReportGenerator()

# ADD THIS JINJA2 FILTER (MISSING IN YOUR CODE)
@app.template_filter('from_json')
def from_json_filter(s):
    """Convert JSON string to Python dict for Jinja2 templates"""
    try:
        return json.loads(s) if s else {}
    except (json.JSONDecodeError, TypeError):
        return {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Dashboard homepage with recent detections"""
    recent_detections = DetectionResult.query.order_by(
        DetectionResult.timestamp.desc()
    ).limit(10).all()
    return render_template('dashboard.html', detections=recent_detections)

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/detection', methods=['POST'])
def detect():
    """Handle file upload and detection"""
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    detection_type = request.form.get('detection_type', 'deepfake')
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        # Perform detection based on type
        result = None
        if detection_type == 'deepfake':
            result = deepfake_detector.detect(filepath)
        elif detection_type == 'object':
            result = object_detector.detect(filepath)
        elif detection_type == 'fraud':
            result = fraud_detector.detect(filepath)
        else:
            return jsonify({'success': False, 'error': 'Invalid detection type'}), 400
        
        # Check for errors in result
        if result.get('prediction') == 'error':
            return jsonify({
                'success': False, 
                'error': result.get('error', 'Detection failed')
            }), 500
        
        # Store in database
        detection_record = DetectionResult(
            user_id=1,  # TODO: Replace with actual user auth
            file_path=filepath,
            detection_type=detection_type,
            media_type=result.get('media_type', result.get('type', 'unknown')),
            result=json.dumps(result),
            confidence=result.get('confidence', 0.0),
            timestamp=datetime.fromisoformat(result.get('timestamp', datetime.now().isoformat())),
            meta=json.dumps(result.get('metadata', {}))
        )
        
        db.session.add(detection_record)
        db.session.commit()
        
        # Generate report if requested - FIXED: Check for 'generate_report' not 'generateReport'
        report_path = None
        report_id = None
        generate_report_requested = request.form.get('generate_report') == 'on'
        
        if generate_report_requested:
            try:
                print(f"Generating court-ready evidence report...")
                
                # Generate court report data
                report_data = report_generator.generate_court_report(detection_record)
                
                # Create PDF
                pdf_path = report_generator.create_pdf_report(report_data, detection_record.id)
                
                # Create EvidenceReport database record
                evidence_report = EvidenceReport(
                    detection_id=detection_record.id,
                    report_type='court_evidence',
                    file_path=pdf_path,
                    generated_at=datetime.now(),
                    report_hash=report_generator.generate_hash(report_data),
                    status='completed'
                )
                
                db.session.add(evidence_report)
                db.session.commit()
                
                report_path = pdf_path
                report_id = evidence_report.id
                
                print(f"Report generated successfully: {pdf_path}")
                
            except Exception as report_error:
                print(f"Report generation failed: {report_error}")
                # Don't fail the entire request if report generation fails
                report_path = None
                report_id = None
        
        return jsonify({
            'success': True,
            'detection_id': detection_record.id,
            'detection_type': detection_type,
            'result': result,
            'report_generated': report_path is not None,
            'report_id': report_id,
            'report_path': report_path
        })
        
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/results')
def results():
    """Show all detection results"""
    detections = DetectionResult.query.order_by(DetectionResult.timestamp.desc()).all()
    return render_template('results.html', detections=detections)

@app.route('/reports')
def reports():
    """Show all evidence reports"""
    reports = EvidenceReport.query.order_by(EvidenceReport.generated_at.desc()).all()
    return render_template('reports.html', reports=reports)


@app.route('/reports/download/<int:report_id>')
def download_report(report_id):
    """Download a specific report PDF by detection ID"""
    # Find the detection record
    detection = DetectionResult.query.get_or_404(report_id)
    
    # Find the associated evidence report
    evidence_report = EvidenceReport.query.filter_by(detection_id=detection.id).first()
    
    if not evidence_report or not evidence_report.file_path:
        return jsonify({'error': 'Report not found'}), 404
    
    if not os.path.exists(evidence_report.file_path):
        return jsonify({'error': 'Report file not found on disk'}), 404
    
    # Generate a nice filename
    filename = f"evidence_report_{detection.detection_type}_{detection.id}.pdf"
    
    return send_file(
        evidence_report.file_path,
        as_attachment=True,
        download_name=filename,
        mimetype='application/pdf'
    )

@app.route('/api/stats')
def api_stats():
    """Get dashboard statistics"""
    total_detections = DetectionResult.query.count()
    deepfake_count = DetectionResult.query.filter_by(detection_type='deepfake').count()
    object_count = DetectionResult.query.filter_by(detection_type='object').count()
    fraud_count = DetectionResult.query.filter_by(detection_type='fraud').count()
    
    return jsonify({
        'total_detections': total_detections,
        'deepfake_detections': deepfake_count,
        'object_detections': object_count,
        'fraud_detections': fraud_count,
        'reports_generated': EvidenceReport.query.count()
    })

@app.route('/api/recent_detections')
def api_recent_detections():
    """Get recent detections"""
    detections = DetectionResult.query.order_by(
        DetectionResult.timestamp.desc()
    ).limit(10).all()
    
    return jsonify([{
        'id': d.id,
        'type': d.detection_type,
        'media_type': d.media_type,
        'confidence': d.confidence,
        'timestamp': d.timestamp.isoformat(),
        'prediction': json.loads(d.result).get('prediction', 'unknown')
    } for d in detections])

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)
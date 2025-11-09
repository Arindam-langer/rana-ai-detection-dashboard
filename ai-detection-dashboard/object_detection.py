import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def detect(self, file_path):
        """Detect objects in image or video"""
        try:
            if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                return self._detect_video(file_path)
            else:
                return self._detect_image(file_path)
        except Exception as e:
            return {
                'error': str(e),
                'confidence': 0.0,
                'prediction': 'error',
                'timestamp': datetime.now().isoformat()
            }

    def _detect_image(self, image_path):
        """Detect objects in image"""
        results = self.model(image_path)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    detection = {
                        'class': self.class_names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': [float(x) for x in box.xyxy[0].tolist()],
                        'center': [float((box.xyxy[0][0] + box.xyxy[0][2])/2), 
                                  float((box.xyxy[0][1] + box.xyxy[0][3])/2)]
                    }
                    detections.append(detection)

        # Determine primary class (highest confidence detection)
        primary_class = "unknown"
        max_confidence = 0.0
        
        if detections:
            # Sort by confidence and get the highest
            sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
            primary_class = sorted_detections[0]['class']
            max_confidence = sorted_detections[0]['confidence']

        return {
            'prediction': primary_class,  # Main class of the image
            'confidence': max_confidence,
            'type': 'image',
            'file_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'detections': detections,
            'object_count': len(detections),
            'all_classes': list(set([d['class'] for d in detections])),  # All unique classes found
            'metadata': {
                'model_version': 'YOLOv8n',
                'detection_method': 'object_detection',
                'primary_object': primary_class
            }
        }

    def _detect_video(self, video_path):
        """Detect objects in video"""
        cap = cv2.VideoCapture(video_path)
        frame_detections = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 30 == 0:  # Process every 30th frame
                results = self.model(frame)
                frame_objects = []

                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            detection = {
                                'class': self.class_names[int(box.cls)],
                                'confidence': float(box.conf),
                                'bbox': [float(x) for x in box.xyxy[0].tolist()]
                            }
                            frame_objects.append(detection)

                frame_detections.append({
                    'frame': frame_count,
                    'objects': frame_objects
                })

            frame_count += 1

        cap.release()

        # Aggregate results
        all_detections = []
        class_counts = {}
        
        for frame_data in frame_detections:
            for detection in frame_data['objects']:
                all_detections.append(detection)
                class_name = detection['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Determine primary class (most frequently detected)
        primary_class = "unknown"
        if class_counts:
            primary_class = max(class_counts, key=class_counts.get)

        unique_classes = set([d['class'] for d in all_detections])
        avg_confidence = float(sum([d['confidence'] for d in all_detections]) / len(all_detections)) if all_detections else 0.0

        return {
            'prediction': primary_class,  # Main class of the video
            'confidence': avg_confidence,
            'type': 'video',
            'file_path': video_path,
            'timestamp': datetime.now().isoformat(),
            'unique_objects': list(unique_classes),
            'total_detections': len(all_detections),
            'class_frequency': class_counts,  # How often each class appears
            'frame_analysis': frame_detections,
            'metadata': {
                'total_frames': frame_count,
                'analyzed_frames': len(frame_detections),
                'model_version': 'YOLOv8n',
                'detection_method': 'video_object_detection',
                'primary_object': primary_class
            }
        }
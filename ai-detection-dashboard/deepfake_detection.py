"""
Lightweight deepfake detection using pretrained models from Hugging Face.
NO TRAINING REQUIRED - uses transfer learning.
"""

import os
from typing import Dict, Union, List, Optional
from datetime import datetime

import numpy as np
import torch
import cv2
from PIL import Image
import torchaudio

from transformers import (
    pipeline,
    AutoFeatureExtractor, 
    AutoModelForAudioClassification,
    AutoImageProcessor,
    AutoModelForImageClassification
)


class DeepfakeDetector:
    """
    Lightweight deepfake detector using pretrained Hugging Face models.
    No training required - works out of the box!
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        image_model_name: str = "Organika/sdxl-detector",
        audio_model_name: str = "mo-thecreator/Deepfake-audio-detection"
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        print(f"Loading models on device: {self.device}")
        
        # Image detection using pretrained Hugging Face model
        try:
            print(f"DEBUG:Loading image model: {image_model_name}")
            self.image_processor = AutoImageProcessor.from_pretrained(image_model_name)
            self.image_model = AutoModelForImageClassification.from_pretrained(image_model_name).to(self.device)
            self.image_model.eval()
            self.image_model_name = image_model_name
            print("DEBUG:Image model loaded successfully!")
        except Exception as e:
            print(f"DEBUG:Image model load failed: {e}")
            self.image_model = None
            self.image_processor = None
            self.image_model_name = None
        
        # Audio detection using pretrained Hugging Face model
        try:
            print(f"🎵 Loading audio model: {audio_model_name}")
            self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained(audio_model_name)
            self.audio_model = AutoModelForAudioClassification.from_pretrained(audio_model_name).to(self.device)
            self.audio_model.eval()
            self.audio_model_name = audio_model_name
            print("DEBUG:Audio model loaded successfully!")
        except Exception as e:
            print(f"DEBUG:Audio model load failed: {e}")
            self.audio_feature_extractor = None
            self.audio_model = None
            self.audio_model_name = None

    # Image Detection
    def detect_image(self, image_path: str) -> Dict[str, Union[str, float]]:
        """Detect if image is deepfake using pretrained model"""
        
        if not os.path.exists(image_path):
            return {
                "error": "file not found",
                "prediction": "error",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "type": "image",
                "media_type": "image",
                "file_path": image_path
            }
        
        if self.image_model is None or self.image_processor is None:
            return {
                "error": "image model not loaded",
                "prediction": "error",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "type": "image",
                "media_type": "image",
                "file_path": image_path
            }
        
        try:
            # Empty cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.image_model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get prediction
                pred_idx = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred_idx].item()
                
                # Map to "real" or "fake"
                label_map = self.image_model.config.id2label
                pred_label = label_map[pred_idx].lower()
                
                # Normalize labels
                if "fake" in pred_label or "deepfake" in pred_label:
                    prediction = "fake"
                else:
                    prediction = "real"
            
            return {
                "prediction": prediction,
                "confidence": float(confidence),
                "type": "image",
                "media_type": "image",
                "file_path": image_path,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "model_version": "pretrained-hf",
                    "model_name": self.image_model_name,
                    "detection_method": "Transfer Learning (No Training Required)",
                    "raw_label": pred_label
                }
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "prediction": "error",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "type": "image",
                "media_type": "image",
                "file_path": image_path
            }

    # Video Detection
    def _sample_frames(self, video_path: str, max_frames: int = 16) -> List[np.ndarray]:
        """Sample frames uniformly from video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return []
        
        # Sample frames uniformly
        indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames

    def detect_video(self, video_path: str, max_frames: int = 16) -> Dict[str, Union[str, float, dict]]:
        """
        Detect deepfake in video by analyzing sampled frames.
        Uses per-frame classification with majority voting.
        """
        
        if not os.path.exists(video_path):
            return {
                "error": "file not found",
                "prediction": "error",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "type": "video",
                "media_type": "video",
                "file_path": video_path
            }
        
        try:
            # Sample frames
            frames = self._sample_frames(video_path, max_frames=max_frames)
            
            if not frames:
                return {
                    "error": "no frames extracted",
                    "prediction": "error",
                    "confidence": 0.0,
                    "timestamp": datetime.now().isoformat(),
                    "type": "video",
                    "media_type": "video",
                    "file_path": video_path
                }
            
            # Analyze each frame
            frame_results = []
            for i, frame in enumerate(frames):
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Save temporarily
                temp_path = f"/tmp/temp_frame_{i}.jpg"
                pil_image.save(temp_path)
                
                # Detect on frame
                result = self.detect_image(temp_path)
                frame_results.append(result)
                
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            # Aggregate results using majority voting
            fake_count = sum(1 for r in frame_results if r.get("prediction") == "fake")
            real_count = len(frame_results) - fake_count
            
            overall_prediction = "fake" if fake_count > real_count else "real"
            
            # Average confidence
            confidences = [r.get("confidence", 0.0) for r in frame_results]
            avg_confidence = float(np.mean(confidences)) if confidences else 0.0
            
            return {
                "prediction": overall_prediction,
                "confidence": avg_confidence,
                "type": "video",
                "media_type": "video",
                "file_path": video_path,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "sampled_frames": len(frames),
                    "fake_count": fake_count,
                    "real_count": real_count,
                    "frame_predictions": [r.get("prediction") for r in frame_results],
                    "model_version": "pretrained-hf",
                    "detection_method": "Per-frame majority voting (No Training Required)"
                }
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "prediction": "error",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "type": "video",
                "media_type": "video",
                "file_path": video_path
            }

    # Audio Detection
    def detect_audio(self, audio_path: str) -> Dict[str, Union[str, float]]:
        """Detect if audio is deepfake using pretrained model"""
        
        if self.audio_model is None or self.audio_feature_extractor is None:
            return {
                "error": "audio model not loaded",
                "prediction": "error",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "type": "audio",
                "media_type": "audio",
                "file_path": audio_path
            }
        
        if not os.path.exists(audio_path):
            return {
                "error": "file not found",
                "prediction": "error",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "type": "audio",
                "media_type": "audio",
                "file_path": audio_path
            }

        try:
            # Empty cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
            
            target_sr = getattr(self.audio_feature_extractor, "sampling_rate", 16000)
            if sr != target_sr:
                waveform = torchaudio.functional.resample(waveform, sr, target_sr)

            samples = waveform.squeeze(0).numpy()
            inputs = self.audio_feature_extractor(
                samples, 
                sampling_rate=target_sr, 
                return_tensors="pt", 
                padding=True
            )
            
            # Move to device
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.audio_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)

            # Get label
            label_str = self.audio_model.config.id2label[pred.item()] if hasattr(self.audio_model.config, "id2label") else str(pred.item())
            label_str = label_str.lower()
            
            # Map to "real" or "fake"
            if "bonafide" in label_str or "real" in label_str:
                pred_label = "real"
            elif "spoof" in label_str or "fake" in label_str:
                pred_label = "fake"
            else:
                pred_label = "fake" if pred.item() == 1 else "real"

            return {
                "prediction": pred_label,
                "confidence": float(conf.item()),
                "type": "audio",
                "media_type": "audio",
                "file_path": audio_path,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "sampling_rate": target_sr,
                    "model_version": "pretrained-hf",
                    "model_name": self.audio_model_name,
                    "detection_method": "Transfer Learning (No Training Required)",
                    "raw_label": label_str
                }
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "prediction": "error",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "type": "audio",
                "media_type": "audio",
                "file_path": audio_path
            }


##check and routing to functions in class according to types
    def detect(self, file_path: str) -> Dict[str, Union[str, float, dict]]:
        """Universal detection method that routes to appropriate detector"""        
        if not os.path.exists(file_path):
            return {
                "error": "file not found",
                "prediction": "error",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "file_path": file_path
            }
        
        ext = os.path.splitext(file_path)[1].lower()
        
        # Route to appropriate detector
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"):
            return self.detect_image(file_path)
        elif ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            return self.detect_video(file_path)
        elif ext in (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"):
            return self.detect_audio(file_path)
        else:
            return {
                "error": "unsupported file type",
                "prediction": "error",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "file_path": file_path
            }
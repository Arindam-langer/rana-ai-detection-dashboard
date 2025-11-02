import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import cv2
import os
from typing import Dict, Union
import numpy as np


class DeepfakeDetector:
    """
    A class to detect deepfakes in images and videos using pre-trained models.
    This detector can analyze both individual images and video files frame-by-frame.
    """

    def __init__(self):
        """
        Sets up the computing device (GPU if available, otherwise CPU).
        Note: Model loading happens when detect_image or detect_video is called.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # we will initialize them as we need because we use two different pre trained models
        # if you find one singular model then good
        self.model = None
        self.processor = None
        self.current_model_name = None

    def load_model(self, model_name: str):
        """
        Load a specific model from Hugging Face.
        Only loads if we haven't already loaded this model.
        Args:
            model_name: The Hugging Face model identifier (e.g., "username/model-name")
        """
        # Only load the model if it's different from what's currently loaded
        if self.current_model_name != model_name:
            try:
                # Load the image processor (handles image preprocessing)
                self.processor = AutoImageProcessor.from_pretrained(model_name)

                # Load the actual classification model and move it to GPU/CPU
                self.model = AutoModelForImageClassification.from_pretrained(
                    model_name
                ).to(self.device)

                # Set model to evaluation mode (disables dropout, batch normalization updates, etc.)
                self.model.eval()

                # Remember which model we loaded
                self.current_model_name = model_name

            except Exception as e:
                # If something goes wrong, raise an error with details
                raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    # detecting which file it is and calling functions accordingly
    """
    Args:
        file_path: Path to the file (image or video)
        
    Returns:
        Dictionary with detection results (format depends on file type)
    """

    def detect(self, file_path: str) -> Dict[str, Union[str, float]]:

        if not os.path.exists(file_path):
            return {"error": "File not found", "prediction": None, "confidence": 0.0}

        # videos
        video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")

        # images
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        #making them lowercase for ease
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext in video_extensions:
            return self.detect_video(file_path)
        elif file_ext in image_extensions:
            return self.detect_image(file_path)

        else:
            return {
                "error": "Unsupported file format",
                "prediction": None,
                "confidence": 0.0,
            }

    def detect_image(self, image_path: str) -> Dict[str, Union[str, float]]:
        """
        How it works:
        1. Loads the image from disk
        2. Preprocesses it (resize, normalize, etc.)
        3. Runs it through the neural network
        4. Returns prediction (fake/real) with confidence score

        Args:
            image_path: Full path to the image file (e.g., "/path/to/image.jpg")

        Returns:
            Dictionary containing:
            - prediction: "fake" or "real"
            - confidence: How confident the model is (0.0 to 1.0)
            - file_path: The path to the analyzed file
            - type: "image"
            - error: Error message if something went wrong
        """
        try:
            # pre-trained model for image deepfakes
            model_name = "dima806/deepfake_vs_real_image_detection"
            self.load_model(model_name)

            if not os.path.exists(image_path):
                return {
                    "error": "File not found",
                    "prediction": None,
                    "confidence": 0.0,
                }

            # convert it to RGB format because of model requirements
            image = Image.open(image_path).convert("RGB")

            # Preprocess the image: resize, normalize pixel values, convert to tensor
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            # Run the model without calculating gradients (saves memory and computation)
            with torch.no_grad():
                outputs = self.model(**inputs)
                preds = torch.softmax(outputs.logits, dim=1)
                conf, label = torch.max(preds, 1)

                # Convert the numeric label to human-readable text (real or fake)
                prediction = self.model.config.id2label[label.item()]

            return {
                "prediction": prediction,
                "confidence": float(conf),
                "file_path": image_path,
                "type": "image",
            }

        except Exception as e:
            return {
                "error": str(e),
                "prediction": None,
                "confidence": 0.0,
                "file_path": image_path,
            }

    def detect_video(
        self, video_path: str, sample_rate: int = 30
    ) -> Dict[str, Union[str, float]]:
        """
        Detect if a video contains deepfake content by analyzing multiple frames.

        How it works:
        1. Opens the video file
        2. Extracts frames at regular intervals (every 30th frame by default)
        3. Analyzes each sampled frame using the deepfake detection model
        4. Aggregates all frame predictions into an overall verdict

        Args:
            video_path: Full path to the video file (e.g., "/path/to/video.mp4")
            sample_rate: Analyze every Nth frame (default: 30)
                        Higher = faster but less accurate
                        Lower = slower but more thorough

        Returns:
            Dictionary containing:
            - prediction: Overall verdict ("fake" or "real")
            - confidence: Average confidence across all analyzed frames
            - file_path: The path to the analyzed video
            - type: "video"
            - metadata: Additional info (total frames, analyzed frames, fps, etc.)
            - error: Error message if something went wrong
        """
        try:
            model_name = "Naman712/Deep-fake-detection"
            self.load_model(model_name)

            if not os.path.exists(video_path):
                return {
                    "error": "File not found",
                    "prediction": None,
                    "confidence": 0.0,
                }
            cap = cv2.VideoCapture(video_path)

            # Check if the video was successfully opened
            if not cap.isOpened():
                return {
                    "error": "Cannot open video file",
                    "prediction": None,
                    "confidence": 0.0,
                }
            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )  
            fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

            # Lists to store predictions and confidences for each analyzed frame
            frame_predictions = []
            frame_confidences = []

            # Counters to keep track of progress
            analyzed_frames = 0  # How many frames we actually analyzed
            current_frame = 0  # Which frame we're currently on

            # Loop through all frames in the video
            while True:
                # Read the next frame
                ret, frame = cap.read()

                # If we couldn't read a frame, we've reached the end of the video
                if not ret:
                    break

                # Only analyze frames at our sample rate (e.g., every 30th frame)
                # This saves time - we don't need to analyze every single frame
                ##fucking math i got from github and chatgpt 
                if current_frame % sample_rate == 0:
                    # OpenCV reads images in BGR format, but our model expects RGB
                    # So we need to convert the color format
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Convert the NumPy array to a PIL Image (required by the processor)
                    pil_image = Image.fromarray(frame_rgb)

                    # Preprocess the frame just like we do for images
                    inputs = self.processor(images=pil_image, return_tensors="pt").to(
                        self.device
                    )
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        preds = torch.softmax(outputs.logits, dim=1)
                        conf, label = torch.max(preds, 1)
                        prediction = self.model.config.id2label[label.item()]
                    frame_predictions.append(prediction)
                    frame_confidences.append(float(conf))
                    analyzed_frames += 1

                current_frame += 1
            cap.release()
            if not frame_predictions:
                return {
                    "error": "No frames analyzed",
                    "prediction": None,
                    "confidence": 0.0,
                }

            fake_count = sum(1 for p in frame_predictions if p.lower() == "fake")

            real_count = len(frame_predictions) - fake_count

            avg_confidence = sum(frame_confidences) / len(frame_confidences)

            if fake_count > real_count:
                overall_prediction = "fake"     
            else:
                overall_prediction = "real"

            # Calculate what percentage of frames were detected as fake
            fake_ratio = fake_count / len(frame_predictions)

            # Return comprehensive results
            return {
                "prediction": overall_prediction,
                "confidence": avg_confidence,
                "file_path": video_path,
                "type": "video",
                "metadata": {
                    "total_frames": total_frames,  # Total frames in video
                    "analyzed_frames": analyzed_frames,  # How many we actually checked
                    "fps": fps,  # Video frame rate
                    "fake_frames": fake_count,  # Frames detected as fake
                    "real_frames": real_count,  # Frames detected as real
                    "fake_ratio": fake_ratio,  # Percentage of fake frames
                },
            }

        except Exception as e:
            # If anything goes wrong, return an error dictionary
            return {
                "error": str(e),
                "prediction": None,
                "confidence": 0.0,
                "file_path": video_path,
            }

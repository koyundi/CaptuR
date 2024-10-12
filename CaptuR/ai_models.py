from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForImageClassification, AutoImageProcessor, AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from scipy.io import wavfile
import torch

class TextAIDetector:
    """AI Model for Text-based Deepfake Detection."""
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("andreas122001/roberta-academic-detector")
        self.model = AutoModelForSequenceClassification.from_pretrained("andreas122001/roberta-academic-detector")

    def predict(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
            predicted_class_id = logits.argmax().item()
        return self.model.config.id2label[predicted_class_id]


class ImageAIDetector:
    """AI Model for Image-based Deepfake Detection."""
    def __init__(self):
        # Load your image model (example only)
        self.processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection") 
        self.model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
    def predict(self, image):
        # Process the image to match the model's input format
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

        with torch.no_grad():
            logits = self.model(pixel_values).logits
            predicted_class_id = logits.argmax(-1).item()

        # Return the predicted label
        return self.model.config.id2label[predicted_class_id]
    
    
class AudioAIDetector:
    """AI Model for Audio-based Deepfake Detection."""
    def __init__(self):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("MelodyMachine/Deepfake-audio-detection-V2")
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained("MelodyMachine/Deepfake-audio-detection-V2")

    def predict(self, wav_file):
        # Read the WAV file and extract the sampling rate and audio data
        sampling_rate, audio_array = wavfile.read(wav_file)

        # Ensure the audio is in the correct tensor format
        audio_array = torch.tensor(audio_array, dtype=torch.float32)

        # Extract features using the feature extractor
        inputs = self.feature_extractor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")

        # Make prediction with the model
        with torch.no_grad():
            logits = self.model(**inputs).logits
            predicted_class_id = torch.argmax(logits, dim=-1).item()

        # Return the predicted label
        return self.model.config.id2label[predicted_class_id]
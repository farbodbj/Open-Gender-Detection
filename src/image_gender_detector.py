import pickle
from common import Gender
import requests
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sklearn.linear_model import SGDClassifier
from logging import getLogger
logger = getLogger(__name__)

class GenderClassifier:
    def __init__(
        self,
        svm_model_path: str,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        uncertainty_threshold: float = 0.6,
    ):
        """
        Initialize the Gender Classifier.

        Args:
            svm_model_path: Path to the saved SVM model pickle file
            clip_model_name: Name of the CLIP model to use (default: "openai/clip-vit-base-patch32")
            uncertainty_threshold: Decision threshold for uncertain classifications (default: 0.6)
                                   Predictions with max probability below this will return Gender.UNKNOWN
        """
        self.svm_model = self._load_svm_model(svm_model_path)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.uncertainty_threshold = uncertainty_threshold

    @staticmethod
    def _load_svm_model(model_path: str) -> SGDClassifier:
        """Load the SVM model from a pickle file."""
        with open(model_path, "rb") as f:
            return pickle.load(f)

    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Generate CLIP embedding for an image.

        Args:
            image: PIL Image object

        Returns:
            numpy array containing the image embedding
        """
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        outputs = self.clip_model.get_image_features(**inputs)
        return outputs.detach().numpy().flatten()

    def predict_gender(self, image: Image.Image) -> Gender:
        """
        Predict gender from an image using the SVM model.

        Args:
            image: PIL Image object

        Returns:
            Gender enum (MALE, FEMALE, or UNKNOWN)
        """
        embedding = self.get_embedding(image)
        predicted_class = self.svm_model.predict([embedding])
        logger.info(f"predicted class by image: {predicted_class}")
        return Gender.MALE if predicted_class == 0 else Gender.FEMALE

    def predict_from_url(self, image_url: str) -> Gender:
        """
        Predict gender from an image URL.

        Args:
            image_url: URL of the image to classify

        Returns:
            Gender enum (MALE, FEMALE, or UNKNOWN)
        """
        image = Image.open(requests.get(image_url, stream=True).raw)
        image.save('image.png')
        return self.predict_gender(image)
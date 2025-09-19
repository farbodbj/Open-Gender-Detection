from .common import Gender
from .name_gender_detector import GenderDetector
from .image_gender_detector import GenderClassifier
from typing import Optional, Tuple
from PIL import Image
import numpy as np

class WeightedGenderPredictor:
    def __init__(
        self,
        svm_model_path: str,
        name_detector: GenderDetector,
        image_weight: float = 0.6,
        name_weight: float = 0.4,
        image_uncertainty_threshold: float = 0.6,
        name_prob_threshold: float = 0.5,
    ):
        """
        Initialize the Weighted Gender Predictor.

        Args:
            svm_model_path: Path to the saved SVM model pickle file for image classification
            name_detector: Instance of GenderDetector for name-based classification
            image_weight: Weight for image classification (default: 0.6)
            name_weight: Weight for name classification (default: 0.4)
            image_uncertainty_threshold: Threshold for uncertain image classifications (default: 0.6)
            name_prob_threshold: Minimum probability threshold for name classification (default: 0.5)
        """
        self.image_classifier = GenderClassifier(
            svm_model_path=svm_model_path,
            uncertainty_threshold=image_uncertainty_threshold
        )
        self.name_detector = name_detector
        self.image_weight = image_weight
        self.name_weight = name_weight
        self.name_prob_threshold = name_prob_threshold

        # Validate weights
        if not np.isclose(image_weight + name_weight, 1.0):
            raise ValueError("Weights must sum to 1.0")

    def _normalize_gender_to_score(self, gender: Gender) -> Tuple[float, float]:
        """
        Convert gender enum to numerical scores for male and female.
        
        Returns:
            Tuple of (male_score, female_score)
        """
        if gender == Gender.MALE:
            return (1.0, 0.0)
        elif gender == Gender.FEMALE:
            return (0.0, 1.0)
        else:
            return (0.0, 0.0)

    def predict_from_combined(
        self,
        image: Optional[Image.Image] = None,
        image_url: Optional[str] = None,
        display_name: Optional[str] = None
    ) -> Tuple[Gender, Optional[float]]:
        """
        Predict gender using weighted voting from both image and name classifiers.

        Args:
            image: PIL Image object (optional)
            image_url: URL of the image to classify (optional)
            display_name: Display name for name-based classification (optional)

        Returns:
            Tuple of predicted Gender enum and combined confidence score (None if unknown)
        """
        image_pred = None
        name_pred = None
        image_score = 0.0
        name_score = 0.0

        # Get image prediction if image is provided
        if image is not None or image_url is not None:
            if image is not None:
                image_pred = self.image_classifier.predict_gender(image)
            else:
                image_pred = self.image_classifier.predict_from_url(image_url)
            
            # Convert prediction to scores
            image_male, image_female = self._normalize_gender_to_score(image_pred)
            image_score = (image_male - image_female) * self.image_weight

        # Get name prediction if name is provided
        if display_name is not None:
            name_gender_str, name_prob = self.name_detector.guess_gender(display_name)
            
            # Convert name detector output to our Gender enum
            if name_gender_str == Gender.MALE.value and name_prob >= self.name_prob_threshold:
                name_pred = Gender.MALE
                name_score = name_prob * self.name_weight
            elif name_gender_str == Gender.FEMALE.value and name_prob >= self.name_prob_threshold:
                name_pred = Gender.FEMALE
                name_score = -name_prob * self.name_weight
            else:
                name_pred = Gender.UNKNOWN

        # Combine results
        if image_pred is not None and name_pred is not None:
            # Weighted voting
            combined_score = image_score + name_score
            confidence = abs(combined_score)
            
            if combined_score > 0:
                return Gender.MALE, confidence
            elif combined_score < 0:
                return Gender.FEMALE, confidence
            else:
                return Gender.UNKNOWN, None
        
        # If only image prediction is available
        elif image_pred is not None:
            if image_pred != Gender.UNKNOWN:
                return image_pred, self.image_weight
            return Gender.UNKNOWN, None
        
        # If only name prediction is available
        elif name_pred is not None:
            if name_pred != Gender.UNKNOWN and name_prob >= self.name_prob_threshold:
                return name_pred, name_prob
            return Gender.UNKNOWN, None
        
        # If no predictions available
        else:
            return Gender.UNKNOWN, None
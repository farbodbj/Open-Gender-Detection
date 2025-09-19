from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import io
from PIL import Image
import uvicorn

# Import your existing modules
from src.name_gender_detector import GenderDetector, DataManager, NameMatcher, SurnameChecker
from src.image_gender_detector import GenderClassifier
from src.common import Gender
from src.aggregated_predictor import WeightedGenderPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Gender Detection API",
    description="API for detecting gender from names, images, or both",
    version="1.0.0"
)

# Pydantic models for request/response
class NameGenderRequest(BaseModel):
    display_name: str

class ImageUrlRequest(BaseModel):
    image_url: str

class CombinedRequest(BaseModel):
    display_name: Optional[str] = None
    image_url: Optional[str] = None

class GenderResponse(BaseModel):
    gender: str
    confidence: Optional[float]
    method: str

class ErrorResponse(BaseModel):
    error: str
    detail: str

# Global variables for models (initialized on startup)
name_detector = None
image_classifier = None
weighted_predictor = None

def initialize():
    """Initialize models on server startup"""
    global name_detector, image_classifier, weighted_predictor
    
    try:
        logger.info("Initializing gender detection models...")
        import os
        print(os.listdir('.'))
        # Initialize name detector
        data_manager = DataManager(
            names_file='./datasets/persian-gender-by-name.csv',
            surnames_file='./datasets/surnames.csv'
        )
        matcher = NameMatcher(data_manager)
        surname_checker = SurnameChecker(data_manager)
        
        name_detector = GenderDetector(
            data_manager=data_manager,
            matcher=matcher,
            surname_checker=surname_checker
        )
        
        # Initialize image classifier
        image_classifier = GenderClassifier(
            svm_model_path="./model/sgd_classifier.pkl",
            uncertainty_threshold=0.6
        )
        
        # Initialize weighted predictor
        weighted_predictor = WeightedGenderPredictor(
            svm_model_path="./model/sgd_classifier.pkl",
            name_detector=name_detector,
            image_weight=0.6,
            name_weight=0.4
        )
        
        logger.info("All models initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        raise e

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Gender Detection API",
        "version": "1.0.0",
        "endpoints": {
            "name_detection": "/predict/name",
            "image_detection": "/predict/image",
            "image_url_detection": "/predict/image-url",
            "combined_detection": "/predict/combined",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "name_detector": name_detector is not None,
            "image_classifier": image_classifier is not None,
            "weighted_predictor": weighted_predictor is not None
        }
    }

@app.post("/predict/name", response_model=GenderResponse)
async def predict_gender_from_name(request: NameGenderRequest):
    """Predict gender from display name"""
    try:
        if not name_detector:
            raise HTTPException(status_code=500, detail="Name detector not initialized")
        
        logger.info(f"Predicting gender for name: {request.display_name}")
        
        gender_str, probability = name_detector.guess_gender(request.display_name)
        
        return GenderResponse(
            gender=gender_str,
            confidence=probability,
            method="name_based"
        )
        
    except Exception as e:
        logger.error(f"Error in name prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/image", response_model=GenderResponse)
async def predict_gender_from_image(file: UploadFile = File(...)):
    """Predict gender from uploaded image file"""
    try:
        if not image_classifier:
            raise HTTPException(status_code=500, detail="Image classifier not initialized")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        logger.info(f"Predicting gender for uploaded image: {file.filename}")
        
        gender = image_classifier.predict_gender(image)
        
        return GenderResponse(
            gender=gender.value,
            confidence=0.6 if gender != Gender.UNKNOWN else None,  # Default confidence
            method="image_based"
        )
        
    except Exception as e:
        logger.error(f"Error in image prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/image-url", response_model=GenderResponse)
async def predict_gender_from_image_url(request: ImageUrlRequest):
    """Predict gender from image URL"""
    try:
        if not image_classifier:
            raise HTTPException(status_code=500, detail="Image classifier not initialized")
        
        logger.info(f"Predicting gender for image URL: {request.image_url}")
        
        gender = image_classifier.predict_from_url(request.image_url)
        
        return GenderResponse(
            gender=gender.value,
            confidence=0.6 if gender != Gender.UNKNOWN else None,  # Default confidence
            method="image_based"
        )
        
    except Exception as e:
        logger.error(f"Error in image URL prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/combined", response_model=GenderResponse)
async def predict_gender_combined(
    request: CombinedRequest,
    file: Optional[UploadFile] = File(None)
):
    """Predict gender using both name and image (weighted combination)"""
    try:
        if not weighted_predictor:
            raise HTTPException(status_code=500, detail="Weighted predictor not initialized")
        
        if not request.display_name and not request.image_url and not file:
            raise HTTPException(
                status_code=400, 
                detail="At least one of display_name, image_url, or image file must be provided"
            )
        
        image = None
        if file:
            # Validate file type
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            # Read and process image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        
        logger.info(f"Combined prediction - Name: {request.display_name}, Image URL: {request.image_url}, File: {file.filename if file else None}")
        
        gender, confidence = weighted_predictor.predict_from_combined(
            image=image,
            image_url=request.image_url,
            display_name=request.display_name
        )
        
        return GenderResponse(
            gender=gender.value,
            confidence=confidence,
            method="combined"
        )
        
    except Exception as e:
        logger.error(f"Error in combined prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    initialize()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
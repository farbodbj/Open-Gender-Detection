# Open-Gender-Detection

A FastAPI-based service that predicts gender from names, images, or both using machine learning models.

## System Overview

The system combines two detection methods:
- **Name-based**: Uses Persian/English name datasets with fuzzy matching
- **Image-based**: Uses CLIP embeddings with SVM classifier
- **Combined**: Weighted voting between both methods (60% image, 40% name)

## Quick Start

### With Docker (Recommended)
```bash
# Start the service
docker-compose up --build

# API available at http://localhost:8000
```

### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
```

## API Endpoints

### Name Detection
```bash
curl -X POST "http://localhost:8000/predict/name" \
  -H "Content-Type: application/json" \
  -d '{"display_name": "Ahmad Rezaei"}'
```

### Image Detection
```bash
# From URL
curl -X POST "http://localhost:8000/predict/image-url" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/photo.jpg"}'

# From file upload
curl -X POST "http://localhost:8000/predict/image" \
  -F "file=@photo.jpg"
```

### Combined Detection
```bash
curl -X POST "http://localhost:8000/predict/combined" \
  -H "Content-Type: application/json" \
  -d '{"display_name": "Sara", "image_url": "https://example.com/photo.jpg"}'
```

## Response Format
```json
{
  "gender": "m",           // "m", "f", or "u" (unknown)
  "confidence": 0.85,      // 0.0 to 1.0
  "method": "combined"     // "name_based", "image_based", or "combined"
}
```

## Configuration

Default weights in combined mode:
- Image: 60%
- Name: 40%

Modify in `main.py` or `app.py` startup section.

## Health Check
```bash
curl http://localhost:8000/health
```

### Citation
```
@misc{bijary2025agenticusernamesuggestionmultimodal,
      title={Agentic Username Suggestion and Multimodal Gender Detection in Online Platforms: Introducing the PNGT-26K Dataset}, 
      author={Farbod Bijary and Mohsen Ebadpour and Amirhosein Tajbakhsh},
      year={2025},
      eprint={2509.11136},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.11136}, 
}
```
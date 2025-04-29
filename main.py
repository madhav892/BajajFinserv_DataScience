from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import pytesseract
from typing import List, Optional, Dict, Any
import re
from pydantic import BaseModel
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lab Report Processor API",
    description="API for extracting lab test information from medical lab reports using OCR",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class LabTest(BaseModel):
    test_name: str
    test_value: str
    bio_reference_range: Optional[str] = None
    lab_test_out_of_range: Optional[bool] = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "test_name": "Hemoglobin",
                "test_value": "14.5",
                "bio_reference_range": "13.0-17.0",
                "lab_test_out_of_range": False
            }
        }

class LabReportResponse(BaseModel):
    lab_tests: List[LabTest]
    is_success: bool
    error: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "lab_tests": [
                    {
                        "test_name": "Hemoglobin",
                        "test_value": "14.5",
                        "bio_reference_range": "13.0-17.0",
                        "lab_test_out_of_range": False
                    }
                ],
                "is_success": True,
                "error": None
            }
        }

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Improved preprocessing for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Try multiple preprocessing approaches
    results = []
    
    # Method 1: Adaptive threshold
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Method 2: OTSU threshold
    _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 3: Simple binary threshold
    _, thresh3 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Save all versions to debug
    cv2.imwrite("static/original.jpg", image)
    cv2.imwrite("static/gray.jpg", gray)
    cv2.imwrite("static/thresh1.jpg", thresh1)
    cv2.imwrite("static/thresh2.jpg", thresh2)
    cv2.imwrite("static/thresh3.jpg", thresh3)
    
    # Return the OTSU threshold (often best for text)
    return thresh2

def extract_tests_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract lab tests from OCR text"""
    lab_tests = []
    
    # Simpler patterns that are more likely to match
    patterns = [
        # Basic pattern: Test Name Value Range
        r'([A-Za-z\s\-]+)\s+([\d\.]+)\s+([0-9\.-]+)',
        
        # Pattern for test name followed by value and range with units
        r'([A-Za-z\s\-]+)[\:\s]+([\d\.]+)[\s\w]*\s+([0-9\.-]+)',
        
        # Pattern for tabular data
        r'([A-Za-z\s\-]+)\s+([\d\.]+)\s+\w+\s+([0-9\.-]+)'
    ]
    
    # Try all patterns
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            test_name, test_value, reference_range = match
            
            # Clean up values
            test_name = test_name.strip()
            test_value = test_value.strip()
            reference_range = reference_range.strip()
            
            # Skip if too short (likely false positive)
            if len(test_name) < 3 or len(test_value) < 1:
                continue
                
            # Calculate if out of range
            out_of_range = False
            if '-' in reference_range:
                try:
                    min_val, max_val = map(float, reference_range.split('-'))
                    value = float(test_value)
                    out_of_range = value < min_val or value > max_val
                except (ValueError, TypeError):
                    pass
            
            lab_tests.append({
                "test_name": test_name,
                "test_value": test_value,
                "bio_reference_range": reference_range,
                "lab_test_out_of_range": out_of_range
            })
    
    return lab_tests

@app.get("/", tags=["Status"])
def root() -> dict[str, str]:
    return {"message": "Lab Report Processor API is running"}

@app.post("/get-lab-tests", response_model=LabReportResponse)
async def extract_lab_tests(file: UploadFile = File(...)):
    try:
        # Set Tesseract path (adjust for your system)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess image
        processed_img = preprocess_image(img)
        
        # OCR with improved configuration
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        logger.info(f"Extracted text:\n{text}")
        
        # Process text to extract tests
        extracted_tests = extract_tests_from_text(text)
        
        # Convert to Pydantic model
        lab_tests = [
            LabTest(
                test_name=test["test_name"],
                test_value=test["test_value"],
                bio_reference_range=test["bio_reference_range"],
                lab_test_out_of_range=test["lab_test_out_of_range"]
            ) for test in extracted_tests
        ]
        
        return LabReportResponse(lab_tests=lab_tests, is_success=True)
    
    except pytesseract.TesseractNotFoundError:
        return LabReportResponse(
            lab_tests=[],
            is_success=False,
            error="Tesseract OCR not found. Please install it and set the correct path."
        )
    except Exception as e:
        logger.error(f"Error processing lab report: {str(e)}", exc_info=True)
        return LabReportResponse(
            lab_tests=[],
            is_success=False,
            error=f"An error occurred: {str(e)}"
        )

@app.get("/test-ocr", tags=["Development"])
async def test_ocr_endpoint():
    """Test endpoint with a sample image (for development)"""
    test_image_path = "sample_lab_report.png"  # Replace with your test image path
    if not os.path.exists(test_image_path):
        return {"error": "Test image not found"}
    
    with open(test_image_path, "rb") as f:
        return await extract_lab_tests(UploadFile(file=f, filename="test.png"))

@app.post("/debug-ocr", tags=["Development"])
async def debug_ocr(file: UploadFile = File(...)):
    """Debug endpoint to see raw OCR text"""
    try:
        # Set Tesseract path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess image
        processed_img = preprocess_image(img)
        
        # Try different OCR configurations
        results = []
        configs = [
            r'--oem 3 --psm 6',
            r'--oem 3 --psm 4',
            r'--oem 3 --psm 11'
        ]
        
        for config in configs:
            text = pytesseract.image_to_string(processed_img, config=config)
            results.append({"config": config, "text": text})
        
        return {"ocr_results": results}
    except Exception as e:
        return {"error": str(e)}
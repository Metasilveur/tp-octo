"""
Google Cloud Function for animal similarity scoring using Gemini AI.

This function processes images uploaded to Cloud Storage and compares them
with a mystery image to determine animal similarity based on multiple
semantic dimensions.
"""

import base64
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import functions_framework
import vertexai
from google.cloud import storage
from vertexai.generative_models import GenerativeModel, Part
from google.cloud import firestore

# Constants
PROJECT_ID = "acn-gcp-octo-sas"
LOCATION = "europe-west9"
MODEL_NAME = "gemini-2.0-flash-001"
DEFAULT_SCORE = 0

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
storage_client = storage.Client()
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel(model_name=MODEL_NAME)
firestore_client = firestore.Client(database='buir-smart-retail-tp')


class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""
    pass


class MessageParsingError(Exception):
    """Custom exception for message parsing errors."""
    pass


class FirestoreConfigError(Exception):
    """Custom exception for Firestore configuration errors."""
    pass


def get_mystery_image_url() -> str:
    """
    Retrieve mystery image URL from Firestore.
    
    Returns:
        str: URL of the mystery image
        
    Raises:
        FirestoreConfigError: If URL retrieval fails
    """
    try:
        doc_ref = firestore_client.collection("mystery-image").document("animal-1")
        doc = doc_ref.get()
        
        if not doc.exists:
            raise FirestoreConfigError("Mystery image document does not exist")
            
        data = doc.to_dict()
        if not data or "url" not in data:
            raise FirestoreConfigError("Mystery image URL not found in document")
            
        return data["url"]
        
    except Exception as e:
        raise FirestoreConfigError(f"Failed to retrieve mystery image URL: {e}") from e


def get_system_prompt() -> str:
    """
    Retrieve system prompt from Firestore.
    
    Returns:
        str: System prompt for animal similarity scoring
        
    Raises:
        FirestoreConfigError: If prompt retrieval fails
    """
    try:
        doc_ref = firestore_client.collection("mystery-image").document("prompt")
        doc = doc_ref.get()
        
        if not doc.exists:
            raise FirestoreConfigError("Prompt document does not exist")
            
        data = doc.to_dict()
        if not data or "value" not in data:
            raise FirestoreConfigError("Prompt not found in document")
            
        return data["value"]
        
    except Exception as e:
        raise FirestoreConfigError(f"Failed to retrieve system prompt: {e}") from e


def parse_pubsub_message(cloud_event: Any) -> tuple[str, str, str]:
    """
    Parse the Pub/Sub message to extract bucket and filename information.
    
    Args:
        cloud_event: The Cloud Event containing the Pub/Sub message
        
    Returns:
        tuple: (bucket_name, filename, image_uri)
        
    Raises:
        MessageParsingError: If message parsing fails
    """
    try:
        message_data = base64.b64decode(
            cloud_event.data["message"]["data"]
        ).decode("utf-8")
        data = json.loads(message_data)
        
        bucket = data["bucket"]
        filename = data["name"]
        image_uri = f"gs://{bucket}/{filename}"
        
        return bucket, filename, image_uri
        
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise MessageParsingError(f"Failed to parse Pub/Sub message: {e}") from e


def get_folder_name(filename: str) -> str:
    """
    Extract folder name from filename path.
    
    Args:
        filename: Full path to the file
        
    Returns:
        str: Folder name (first part of the path)
    """
    return filename.split("/")[0] if "/" in filename else ""


def generate_similarity_score(image_uri: str, mystery_image_url: str, system_prompt: str) -> int:
    """
    Generate similarity score between uploaded image and mystery image.
    
    Args:
        image_uri: URI of the uploaded image
        mystery_image_url: URL of the mystery image
        system_prompt: System prompt for comparison
        
    Returns:
        int: Similarity score (0-100)
        
    Raises:
        ImageProcessingError: If image processing fails
    """
    try:
        # Create content parts for both images
        content_parts = [
            Part.from_uri(image_uri, mime_type="image/jpeg"),
            Part.from_uri(mystery_image_url, mime_type="image/jpeg"),
            system_prompt
        ]
        
        # Generate content using Gemini
        result = model.generate_content(content_parts)
        
        # Parse score from response
        try:
            score = int(result.text.strip())
            if not (0 <= score <= 100):
                logger.warning(f"Score {score} outside valid range, defaulting to {DEFAULT_SCORE}")
                return DEFAULT_SCORE
            return score
            
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse score from response: {result.text}. Error: {e}")
            return DEFAULT_SCORE
            
    except Exception as e:
        raise ImageProcessingError(f"Failed to process images: {e}") from e


@functions_framework.cloud_event
def process_pubsub_message(cloud_event: Any) -> None:
    """
    Process Pub/Sub message containing image upload information.
    
    This function is triggered when an image is uploaded to Cloud Storage.
    It compares the uploaded image with a mystery image and generates
    a similarity score.
    
    Args:
        cloud_event: The Cloud Event containing the Pub/Sub message
    """
    try:
        # Parse the message
        bucket, filename, image_uri = parse_pubsub_message(cloud_event)
        folder_name = get_folder_name(filename)
        
        logger.info(f"Processing image: {image_uri} (folder: {folder_name})")
        
        # Retrieve configuration from Firestore
        mystery_image_url = get_mystery_image_url()
        system_prompt = get_system_prompt()
        
        logger.info(f"Using mystery image: {mystery_image_url}")
        
        # Generate similarity score
        score = generate_similarity_score(image_uri, mystery_image_url, system_prompt)
        
        logger.info(f"Generated similarity score: {score}")
        
        # Enregistrement dans Firestore
        doc_ref = firestore_client \
            .collection("users") \
            .document(bucket) \
            .collection("predictions") \
            .document(filename)

        doc_ref.set({
            "file_name": filename,
            "inference": score,
            "timestamp": datetime.now(timezone.utc),
            "gcs_uri": image_uri,
            "user": bucket
        })

        logger.info(f"Prediction saved in Firestore for user {folder_name}, image {filename}")

        
    except MessageParsingError as e:
        logger.error(f"Message parsing error: {e}")
    except ImageProcessingError as e:
        logger.error(f"Image processing error: {e}")
    except FirestoreConfigError as e:
        logger.error(f"Firestore configuration error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error processing message: {e}")

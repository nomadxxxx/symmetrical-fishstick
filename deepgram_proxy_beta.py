from fastapi import FastAPI, HTTPException, Request, UploadFile, Form, Depends, File  # Added File Import
import httpx
from httpx import Timeout
import os
import sys
from dotenv import load_dotenv
import logging
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from typing import Optional, Dict, Any
import io
import asyncio  # Import the asyncio library
from fastapi.middleware.cors import CORSMiddleware
from fastapi import status
from starlette.middleware import Middleware
from starlette.responses import StreamingResponse
from starlette.background import BackgroundTasks

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
origins = ["*"]  # Allow all origins - be cautious in production!
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    load_dotenv()
    logger.info(".env file loaded successfully.")
except Exception as e:
    print(f"Error loading .env file: {e}")  # Print is deliberate as logging may not have been set up
    sys.exit(1)  # Exit if .env loading fails - critical for configuration

# Deepgram API Key
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    logger.critical("Deepgram API key is missing. Please set DEEPGRAM_API_KEY in .env file.")
    sys.exit(1)  # Exit if API key is missing - critical for operation

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

# Model Mapping: OpenWebUI -> Deepgram
OPENWEBUI_TO_DEEPGRAM_MODELS = {
    "whisper-small": "nova-2",  # Example mapping
    "whisper-medium": "nova-2",  # Example mapping
    "whisper-large": "nova-2",
    "faster-whisper": "nova-2",  # If you want to alias to the same model.
    "nova-2": "nova-2"  # The user has selected the Deepgram model
}

# Retry Logic
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5), retry=retry_if_exception_type(httpx.HTTPStatusError), reraise=True)
async def make_deepgram_request(
    url: str, headers: Dict[str, str], audio_data: bytes, params: Dict[str, Any]
) -> httpx.Response:
    """Sends a request to the Deepgram API with retry logic."""
    async with httpx.AsyncClient(timeout=Timeout(30.0)) as client:
        logger.info(f"Sending request to Deepgram: url={url}, params={params}")
        logger.debug(f"Deepgram request headers: {headers}")  # Log the headers

        try:
            response = await client.post(url, headers=headers, content=audio_data, params=params)
            response.raise_for_status()  # Raise HTTPStatusError for retry
            return response
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Deepgram API request failed: {e.response.status_code} - {e.response.text}"
            )
            raise  # Re-raise the exception for tenacity to handle
        except httpx.RequestError as e:
            logger.error(f"Request error occurred: {e}")
            raise
        except Exception as e:
            logger.exception("An unexpected error occurred during the request")
            raise


# Rewrites any /v1/audio/transcriptions requests to /v1 internally.
@app.middleware("http")
async def rewrite_path(request: Request, call_next):
    if request.url.path == "/v1/audio/transcriptions":
        scope = request.scope
        scope["path"] = "/v1"  # Rewrite the path
        request = Request(scope, request.receive)  # Create a new request object

    return await call_next(request)

@app.api_route("/v1", methods=["POST"])
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    content_type: Optional[str] = Form(None),  # Add content_type as an optional form parameter
):
    """
    Handles POST requests for audio transcription.
    For POST, expects a multipart/form-data request with an audio file.
    """

    logger.info(f"Received request with method: {request.method}")

    try:
        if request.method == "POST":
            logger.info(f"Content-Type: {request.headers.get('content-type')}")

            # *** INSPECTION: Print audio_file information ***
            logger.info(f"Audio File: {file}")
            logger.info(f"Audio File Filename: {file.filename}")
            logger.info(f"Audio File Content Type: {file.content_type}")
            # **********************************************

            audio_content = await file.read()
            #Use the content type
            #content_type = file.content_type  # Get content type from UploadFile

            logger.info(f"Audio file Content-Type: {content_type}")

            #OpenWebUI requires the models be hardcoded.
            #model = "whisper-large"
            language = "en"

            # Prepare parameters for Deepgram
            deepgram_model = OPENWEBUI_TO_DEEPGRAM_MODELS.get(model)  # Use .get for safety
            if not deepgram_model:
                logger.warning(
                    f"OpenWebUI model '{{model}}' not found in mapping. Using default 'nova-2'."
                )
                deepgram_model = "nova-2"  # Or another suitable Deepgram default
            else:
                logger.info(f"Mapping OpenWebUI model '{{model}}' to Deepgram model '{{deepgram_model}}')")

            params = {"model": deepgram_model, "language": language}  # Use mapped model

            # Set a default Content-Type if it's None
            if content_type is None:
                content_type = "audio/wav"  # Or any other suitable default
                logger.warning("Audio Content-Type not set. Using default 'audio/wav'. Please configure in OpenWebUI settings.")


            # Prepare headers for Deepgram
            #Add token
            headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": content_type}

            # *** INSPECTION: Print request headers and body ***
            logger.info(f"Request Headers: {request.headers}")
            #Try to decode body if possible
            try:
                body = await request.body()
                logger.info(f"Request Body: {body.decode()}")
            except:
                logger.info(f"Request Body: Could not decode")
            # **********************************************

            # Call Deepgram API
            try:
                deepgram_data = audio_content #Corrected line
                deepgram_response = await make_deepgram_request(
                    DEEPGRAM_URL, headers, deepgram_data, params
                )
            except Exception as e:
                logger.error(f"Deepgram Request failed after retries {e}")
                raise HTTPException(status_code=500, detail=f"Deepgram API error: {e}") from e

        else:
            raise HTTPException(status_code=405, detail="Method not allowed")

        # Extract transcription text
        try:
            deepgram_data = deepgram_response.json()
            transcription_text = (
                deepgram_data.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
            )
            if not transcription_text:
                raise KeyError("Missing expected keys in Deepgram response")
                logger.info("Transcription success")
        except (KeyError, ValueError) as e:
            logger.error(f"Error extracting transcription: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to extract transcription from Deepgram response."
            ) from e

        # Format the response to match the OpenAI Whisper API format
        openai_response = {"text": transcription_text}
        return openai_response

    except httpx.HTTPStatusError as e:
        logger.error(f"Deepgram API returned an HTTP error: {e}")
        raise HTTPException(
            status_code=e.response.status_code, detail=f"Deepgram API error: {e}") from e
    except Exception as e:
        logger.exception("An unexpected error occurred")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        ) from e

#Add validation error
from fastapi import Request, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation Error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors()}),
    )

# Startup Event
#Fix deprecation warning by migrating to lifespan
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks
    logger.info("Deepgram Proxy service started successfully.")
    yield
    # Shutdown tasks (if any)

app = FastAPI(lifespan=lifespan)

import uvicorn

if __name__ == "__main__":
    logger.info("Starting Uvicorn...")
    try:
        uvicorn.run("deepgram_proxy:app", host="0.0.0.0", port=5001, reload=True)
        print("running")
    except Exception as e:
        logger.critical(f"Uvicorn failed to start: {e}")
        sys.exit(1)  # Crash the proxy if Uvicorn fails to start.

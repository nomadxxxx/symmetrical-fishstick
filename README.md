DeepgramProxy is a configurable proxy server designed to integrate Deepgram's speech-to-text capabilities into applications that natively support the OpenAI Whisper API, notably [Open WebUI](https://github.com/open-webui/open-webui). Key features include:

- API Translation: It accepts POST requests to a /v1 endpoint mimicking the OpenAI Whisper API and rewrites /v1/audio/transcriptions to /v1.

- Model Mapping: It maps Open WebUI's STT_MODEL setting (e.g., whisper-large) to a corresponding Deepgram model (e.g., nova-2) using a predefined dictionary, allowing users to select familiar Whisper model names while leveraging Deepgram's engine.

- Authentication: It securely manages Deepgram API authentication by retrieving the DEEPGRAM_API_KEY from an environment variable and including it in the Authorization header of requests to the Deepgram API.

- Content Type Handling: It accepts an optional content_type parameter from Open WebUI, allowing control over the audio format sent to Deepgram. If none is provided, it defaults to audio/wav.

- Error Handling: It incorporates comprehensive error handling, including retries for transient API errors, detailed logging, and informative HTTP exception responses to the client.

**HOW TO INSTALL:**

`python3 -m venv venv`

`source venv/bin/activate`

`pip install -r requirements.txt
`
Edit the .env and include your [Deepgram](https://deepgram.com/) API key.

Now run the Proxy:
`python3 deepgram_proxy_beta.py`

**HOW TO USE WITH OPEN WEB UI**
Go to setting within Open WebUI, then admin settings, click audio. For Speech-to-Text Engine select: OpenAI. Replace the URL with [](http://localhost:5001/v1), for API simply type `dummy` and for STT Model write `whisper-large`



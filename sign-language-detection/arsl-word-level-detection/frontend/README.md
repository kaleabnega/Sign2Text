# Frontend Instructions

Static browser client for the Arabic word-level sign recognizer. The workflow:

1. Capture webcam frames via `navigator.mediaDevices`.
2. Run **MediaPipe Hands** in the browser (WebGL) to extract 3D landmarks per hand.
3. Normalize and flatten coordinates to match the training pipeline (126 features per frame).
4. Maintain a rolling buffer of 32 frames.
5. POST `{ sequence: buffer }` to the FastAPI `/predict` endpoint.

## Requirements

- Backend running locally:
  ```bash
  uvicorn arsl-word-level-detection.api_server:app --reload
  ```
- If serving frontend from another origin, enable CORS on the FastAPI app.

## Local Preview

```bash
cd project/sign-language-detection/arsl-word-level-detection/frontend
python -m http.server 4173
```

Open `http://localhost:4173`, allow webcam access, and keep the backend URL field as `http://localhost:8000/predict`. Once 32 frames have been collected the UI displays predictions and confidence scores returned by the backend.

## Deployment Notes

- Host `index.html`, `styles.css`, and `main.js` on any static host (Netlify, GitHub Pages, etc.).
- Update the default backend URL to your deployed FastAPI origin.
- Ensure HTTPS for both frontend and backend.

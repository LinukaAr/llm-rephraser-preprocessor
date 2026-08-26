# Rephraser pre-processor

This small package contains a pilot `Constraint` datatype, a negation detector,
retrieval-based rephrasing, a deterministic generative rephraser, a similarity
verifier, and an end-to-end pipeline with latency instrumentation.

### Flow
![Flow](docs/flow.gif)

### Demo video
<a href="https://youtu.be/Xpq4oY3GHPg" target="_blank">
  <img src="docs/thumbnail.png" alt="Demo video">
</a>

Quick start:

1. Create a Python venv and install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If you want the rephraser to call Gemini, also set your API key in PowerShell:

```powershell
$env:GEMINI_API_KEY = "your_api_key_here"
```

The Gemini-backed generator uses `google-genai`. If the SDK, key, or API call is
unavailable, the pipeline fails closed with a visible `fallback_reason` instead
of inventing a rewrite.

You can also place the key in `config/gemini.env`; the generator will load that
file automatically if `GEMINI_API_KEY` is not already present in the shell.

2. Run the pipeline from the CLI:

```powershell
python run_pipeline.py "Don't use jargon in the explanation."
```

3. Run a quick interactive check:

```python
from rephraser.detector import detect_negation
print(detect_negation("Don't use jargon in the explanation."))
```

Pipeline configuration lives in `config/pipeline_config.json`. The default
pipeline keeps a timing trace for `detect_negation`, candidate generation,
similarity scoring, and total runtime.
For this pilot, the similarity threshold is tuned to `0.5` so short
source-preserving affirmative rewrites are accepted instead of being rejected
as too dissimilar.

To switch to the generative rewrite path, change `generation_strategy` to
`generative` in `config/pipeline_config.json`.

The current rewrite templates prefer minimal source-preserving edits, so the
positive candidate keeps as much of the original sentence shape as possible.

For Cloud Run, the hosted default uses the lightweight `token-overlap`
similarity model so the service stays under the memory limit. If you want to
switch back to Qwen embeddings locally, set `SIMILARITY_MODEL=Qwen/Qwen3-Embedding-0.6B`.

The hosted API does not require local setup on your machine if you deploy the
Docker image. The runtime inputs are the container image and optional
environment variables such as `GEMINI_API_KEY` and `PIPELINE_CONFIG_PATH`.

## Host it remotely

The repo now includes a small FastAPI service in [app.py](app.py) and a
container definition in [Dockerfile](Dockerfile).

Cloud Run is the intended host. The container already listens on the platform
`PORT` variable, so you can deploy it directly.

```powershell
docker build -t negation-rephraser .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key_here negation-rephraser
```

Then call the API:

```powershell
curl -X POST http://localhost:8000/rephrase -H "Content-Type: application/json" -d '{"text":"Don't use jargon in the explanation."}'
```

Available routes:

- `GET /health` for a simple health check
- `POST /rephrase` for the hosted pipeline
- `GET /` for a minimal service summary

## Deploy to Cloud Run

1. Set your Google Cloud project and region.

```powershell
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region YOUR_REGION
```

2. Build and deploy from source. Cloud Build will build the container from the
Dockerfile and push it for you.

```powershell
gcloud run deploy negation-rephraser --source . --allow-unauthenticated --port 8000 --memory 1Gi --set-env-vars GEMINI_API_KEY=your_key_here,SIMILARITY_MODEL=token-overlap
```

You can also run the helper script in the repo root:

```powershell
.\deploy-cloudrun.ps1 -ProjectId YOUR_PROJECT_ID -Region YOUR_REGION -GeminiApiKey YOUR_GEMINI_KEY
```

3. If you want to use a custom config file, upload it into the image or mount it
through your build process, then set `PIPELINE_CONFIG_PATH` to that path in
Cloud Run.

4. After deployment, Cloud Run will give you a service URL. Test it with:

```powershell
curl https://YOUR_SERVICE_URL/health
curl -X POST https://YOUR_SERVICE_URL/rephrase -H "Content-Type: application/json" -d '{"text":"Don't use jargon in the explanation."}'
```

## Deploy from GitHub

Yes, you can deploy from GitHub instead of doing it manually from your machine.
The repo now includes a GitHub Actions workflow at [.github/workflows/deploy-cloudrun.yml](.github/workflows/deploy-cloudrun.yml).

Required GitHub secrets:

- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_SERVICE_ACCOUNT`
- `GCP_PROJECT_ID`
- `GCP_REGION`
- `CLOUD_RUN_SERVICE`
- `GEMINI_API_KEY` if you want the hosted service to call Gemini

After the secrets are set, you can:

- push to `main` and let the workflow deploy automatically
- run it manually from GitHub Actions
- trigger it with the GitHub CLI using `gh workflow run deploy-cloudrun.yml`

If you want the workflow to use a different branch name or service name, edit the file directly.

## Minimal Cloud Run checklist

- Keep `Dockerfile` in the repo root.
- Keep the app binding to `0.0.0.0` and the Cloud Run `PORT`.
- Put secrets like `GEMINI_API_KEY` in Cloud Run environment variables, not in the repo.
- Use the `/health` route for health checks and the `/rephrase` route for the actual API.

# See-through decompose service

A standalone GPU HTTP service that wraps [See-through](https://github.com/shitagaki-lab/see-through):
`POST /decompose` an image → poll the job → download a layered `.psd`. The image2live2d client
(`decompose.from_service`) turns that into a rig. See [Windows setup](../../README.windows.md).

## Files
- `app.py` — the FastAPI service (shells out to See-through's `inference_psd.py`)
- `startup.sh` — GCP VM bootstrap (installs See-through + venv, registers a systemd unit)
- `Dockerfile` — reproducible alternative (CUDA 12.8 base)
- `requirements-service.txt` — service-only deps (fastapi + uvicorn)

## Run locally on a GPU box (quick)
```bash
git clone https://github.com/shitagaki-lab/see-through /opt/see-through
cd /opt/see-through && ln -sf common/assets assets
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt fastapi 'uvicorn[standard]'
SEE_THROUGH_DIR=/opt/see-through uvicorn app:app --host 0.0.0.0 --port 8000
```

## Endpoints
- `GET /health` → `{"ok": true, ...}`
- `POST /decompose` (raw image bytes; optional `X-Auth-Token`) → HTTP 202 with `job_id`
- `GET /jobs/{id}` → status, telemetry, error code
- `GET /jobs/{id}/result` → completed `.psd` bytes

Notes: single GPU ⇒ one active request; additional requests receive HTTP 429. Install models first.
Set `SEE_THROUGH_TOKEN` to require auth. Don't expose the port to `0.0.0.0/0` — restrict to your IP.

## Memory profile and diagnostics

See [MEMORY.md](MEMORY.md) for the pinned-source analysis, staged BF16 loading,
structured telemetry, failure codes, and RTX 4060 validation limits.
`SEE_THROUGH_TIMEOUT` defaults to 900 seconds for the entire subprocess workflow;
`SEE_THROUGH_MEMORY_PROFILE` defaults to `staged` (`low_vram` opts into 768px/20-step LayerDiff with alternating UNet/VAE placement; `legacy` retains the original CLI).
Use one Uvicorn worker. Copy **all `service/seethrough/*.py` files** together for a VM
installation; the Dockerfile does this automatically. The adapter targets the Dockerfile's
pinned upstream revision, so pin that revision for manual installs too.

The current API is asynchronous: POST `/decompose` returns HTTP 202 with `job_id`,
GET `/jobs/{id}` returns status/telemetry/error_code, and GET `/jobs/{id}/result`
streams the completed PSD. A busy GPU returns HTTP 429 before reading another upload.

For RTX 4060 8GB, see the [low_vram configuration and validation guide](MEMORY.md#low_vram-rtx-4060-8gb를-위한-선택형-프로필). Real GPU fit, speed and quality remain **unverified**; staged stays the default.

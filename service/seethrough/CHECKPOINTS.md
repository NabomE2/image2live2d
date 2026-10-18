# Stage checkpoints and resume

`staged` and `low_vram` now commit LayerDiff, Depth and PSD separately. Depth
inference parameters, 20-frame conditioning/attention and output processing are
unchanged. This changes orchestration, not the model. RTX 4060 end-to-end execution
and performance remain unverified in this CPU development environment.

## Use from the app

Rebuild both containers with `docker compose --profile gpu up -d --build`.
The pipeline panel shows LayerDiff/Depth/PSD pending, running, done or error,
elapsed times, and which completed stages were reused. Depth remains marked running
until its subprocess finishes; there is no invented percentage for its internal loop.

After a failure, **Resume completed stages** validates and reuses completed snapshots.
If LayerDiff is complete and Depth failed, only Depth and PSD execute. **Retry Depth,
then rebuild PSD** deliberately invalidates Depth and PSD, retaining a valid LayerDiff.
An invalid LayerDiff prevents an explicit Depth restart; normal resume rebuilds the
invalid stage and its descendants. Repeating resume on a complete valid job reuses
all three GPU-service stages and regenerates the local rig/export artifacts.

The browser remembers the latest app job ID. On app restart, the source and remote
job receipt are restored; press resume to reconstruct the local rig from the remote
PSD/checkpoints. No GPU stage is silently restarted after a transient transport error.
Older services without the checkpoint protocol retain their existing 404 retry behavior.

Pre-upgrade temporary outputs have no committed manifests and are not automatically
trusted or imported. The first run under this version creates reusable checkpoints.

## Independent deadlines

| Setting | Compose default | Direct service default |
| --- | --- | --- |
| `SEE_THROUGH_LAYERDIFF_TIMEOUT` | 1800 seconds | `SEE_THROUGH_TIMEOUT` |
| `SEE_THROUGH_DEPTH_TIMEOUT` | 1800 seconds | `SEE_THROUGH_TIMEOUT` |
| `SEE_THROUGH_PSD_TIMEOUT` | 900 seconds | `SEE_THROUGH_TIMEOUT` |
| `SEE_THROUGH_TIMEOUT` | 900 seconds | 900 seconds |

Each subprocess gets a fresh deadline when it starts. Time spent by LayerDiff no
longer consumes Depth's limit. Snapshot validation/copying and model fingerprinting
happen outside the subprocess deadline; they can add disk I/O time. Existing process
group termination still kills a timed-out stage and its descendants.

`legacy` keeps its original whole-process timeout and does not support stage resume.
The default client polling budget follows the server's sum of stage limits plus
600 seconds for validation/transfers. `SEE_THROUGH_CLIENT_TIMEOUT` in Compose
(`IMAGE2LIVE2D_DECOMPOSE_TIMEOUT` in the app) explicitly overrides it. Leave it empty
for automatic budgeting. A client polling timeout does not cancel the remote job;
resume reconnects to a still-running job rather than creating duplicate inference.

For example, set only `SEE_THROUGH_DEPTH_TIMEOUT=2400` to change the Depth execution
budget. Timeout changes do not invalidate completed image results. This is not a
claim that a longer timeout resolves the underlying Depth performance bottleneck.

## API

- `POST /decompose` with raw image bytes returns `{job_id, checkpoint_protocol: 1}`.
- `GET /jobs/{id}` adds `stages`, `stage_timeouts`, `resumable` and `expires_at` to
  the existing status/error/telemetry response.
- `POST /jobs/{id}/resume` validates the chain and continues to PSD.
- `POST /jobs/{id}/resume?stage=depth` forces Depth and then PSD, reusing LayerDiff.
- `POST /jobs/{id}/resume?stage=depth&stop_after=depth` runs **only Depth**, leaving
  the job `paused`. A subsequent plain resume produces PSD without rerunning Depth.
- `GET /jobs/{id}/result` retains the layered PSD response contract.
- Local web proxy: `POST /api/jobs/{id}/resume` or `?stage=depth`. It uses the stored
  remote job receipt, then continues the existing mesh/rig/Live2D pipeline.

Resume requires the configured `X-Auth-Token`, just like upload. A running GPU job
returns 429; overlapping retries never launch another worker. Use one Uvicorn worker
and one service container per job volume/GPU. Multi-process/distributed writers are
not supported. The local web resume action requires the app's same-origin header.

## Why a snapshot is safe to reuse

The pinned upstream CLI writes files incrementally:

1. LayerDiff writes `input/src_img.png`, body PNGs and optional refined head PNGs.
2. Depth writes part `*_depth.png`, then `input/info.json` and `reconstruction.png`.
3. PSD creates `input/optimized/` and writes `input.psd`, `input_depth.psd` and
   `input.psd.json` beside the input directory.

Every stage runs in a new `attempt-*` directory seeded by a **copy**, not hard links,
of the prior committed stage. A failure can therefore never overwrite its source
checkpoint. Success requires process exit 0, expected part/metadata outputs, readable
PNGs and PSDs, followed by SHA-256 hashing of every output. Payload files/directories
are synced, the directory is renamed, then an fsynced manifest is atomically replaced.
Only manifest-referenced completed snapshots can be reused. Orphaned/partial directories
are not completion evidence. Failed attempts and unreferenced snapshots are removed.
A crash after rename but before publication can lose reuse of that stage, never publish
an incomplete stage as successful.

Reuse fingerprints include input content, effective CLI/profile, service/upstream
source/config content, relevant package versions and **model file contents**. Keys
chain LayerDiff -> Depth -> PSD, so invalidated ancestors invalidate their descendants.
Output hashes are rechecked on every resume. Changing only Depth model contents
invalidates Depth/PSD; general command/source changes conservatively invalidate more.
Model hashing streams files without loading weights into tensor memory and may take
time on slow disks. Do not modify model directories while a job runs.

Managed local models are the recommended path. For unmanaged repositories, already
cached Hugging Face snapshots are resolved and pinned to local paths. An unresolved
remote model name gets a nonreusable identity: a mutable repository name alone must
not authorize reuse. Explicit Depth retry then refuses an unverifiable predecessor;
normal resume recomputes it. No cross-job/input deduplication is performed.

## Storage and expiry

Compose mounts `inference-jobs:/jobs` and `app-jobs:/jobs`; these survive container
recreation. Model weights remain in the separate `models` volume. Defaults:

- Service `SEE_THROUGH_JOB_DIR=/jobs` in Compose (`/tmp/seethrough-jobs` otherwise).
- App `IMAGE2LIVE2D_JOB_DIR=/jobs` in Compose (`/tmp/image2live2d-jobs` otherwise).
- `SEE_THROUGH_RETENTION_SECONDS=604800` (7 days), also passed to the app as
  `IMAGE2LIVE2D_RETENTION_SECONDS`.

Inactive service jobs expire after their last completed/failed/paused attempt. The
app expires inactive records after their last persisted update. Active jobs are never
removed by cleanup. Cleanup runs at startup, on new submissions and periodically
(at most hourly). Expiry removes source images, snapshots, metadata and local exports;
GET then returns unknown job, and a new explicit upload is required. Polling does not
extend retention. Budget disk space for three complete stage snapshots plus an active
copy; cleanup is age-based, not an automatic free-space quota. Disk/write failures
remain errors and cannot publish a successful checkpoint.

`docker compose down` preserves the volumes. **`docker compose down -v` deletes
models and all retained jobs.** Download final artifacts before expiry/deletion.

## Verification boundaries

CPU tests simulate partial Depth writes, timeout/restart, corruption, dependency
changes, interrupted manifest publication, independent clocks, duplicate retry,
service/app recovery, expiry, client receipt reuse and UI retry/status transitions.
They serialize real PNG/PSD fixtures but do not run pretrained GPU inference.
Existing tests cover staged/low_vram tensors and PSD/Live2D export behavior. Hardware
validation still needs a real input, completed LayerDiff, interrupted Depth, and a
resume trace proving LayerDiff was reused while Depth received its own full deadline.

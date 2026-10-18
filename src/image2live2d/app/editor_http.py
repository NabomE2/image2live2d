"""HTTP adapter for the editor; existing generation endpoints remain separate."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .editor import Conflict, EditorStore


def store():
    from .server import _web_root

    return EditorStore(os.environ.get("IMAGE2LIVE2D_EDITOR_DIR", _web_root() / "editor-projects"))


def dispatch(handler, method):
    url = urlparse(handler.path)
    segments = url.path.strip("/").split("/")
    create = (len(segments) == 4 and segments[:2] == ["api", "jobs"]
              and segments[3] == "editor-project")
    if url.path in {"/editor", "/editor.js"} and method == "GET":
        file = "editor.html" if url.path == "/editor" else "editor.js"
        handler._send(200, Path(__file__).with_name(file).read_bytes(),
                      "text/html; charset=utf-8" if file.endswith("html") else "text/javascript")
        return True
    if not create and segments[:3] != ["api", "editor", "projects"]:
        return False
    try:
        data = {}
        if method == "POST":
            if handler.headers.get("Origin") != "http://" + handler.headers.get("Host", ""):
                handler._json(403, {"error": "Use the editor in this app"})
                return True
            length = int(handler.headers.get("Content-Length", "0"))
            if not 0 <= length <= 100_000:
                handler._json(413, {"error": "Editor request too large"})
                return True
            data = json.loads(handler.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
        db = store()
        if create:
            if method != "POST":
                raise ValueError("Create requires POST")
            from .server import get_job

            job = get_job(segments[2])
            if not job or job.status != "done" or job.rig is None:
                raise Conflict("A completed in-memory generation job is required")
            with job.live2d_lock:
                # Use the unlocked builder while holding the same lock as resume/export.
                from .server import _build_live2d_bundle

                bundle = _build_live2d_bundle(job)
                if bundle is None:
                    raise Conflict("Model is not ready")
                result = db.create(job.rig, Path(job.stack.layers[0].texture_path).parent,
                                   bundle, job.source, job.filename, job.work / "normalized")
            handler._json(201, result)
            return True
        pid = segments[3] if len(segments) > 3 else None
        action = segments[4] if len(segments) == 5 else ""
        if len(segments) > 5:
            raise ValueError("Unknown editor route")
        if method == "GET":
            query = parse_qs(url.query)
            if not pid:
                handler._json(200, db.list())
            elif action == "asset":
                path = db.asset(pid, query["part"][0],
                                revision=query.get("revision", [None])[0],
                                draft=query.get("draft", [None])[0])
                handler._send(200, path.read_bytes(), "image/png")
            elif action == "download":
                body = db.download(pid, query["revision"][0])
                handler.send_response(200)
                handler.send_header("Content-Type", "application/zip")
                handler.send_header("Content-Disposition", 'attachment; filename="edited_vtube.zip"')
                handler.send_header("Content-Length", str(len(body)))
                handler.end_headers()
                handler.wfile.write(body)
            elif not action:
                handler._json(200, db.info(pid))
            else:
                raise ValueError("Unknown editor route")
        elif method == "POST":
            if not pid:
                raise ValueError("Project required")
            if action == "drafts":
                result = db.draft(pid, data["base"], data["part_id"])
            elif action == "strokes":
                result = db.strokes(pid, data["draft"], data["sequence"], data["strokes"])
            elif action == "apply":
                result = db.apply(pid, data["draft"], data["sequence"], data.get("approved"))
            elif action == "restore":
                result = db.restore(pid, data["target"], data["base"], data.get("approved"))
            else:
                raise ValueError("Unknown editor action")
            handler._json(200, result)
    except Conflict as exc:
        handler._json(409, {"error": str(exc)})
    except FileNotFoundError:
        handler._json(404, {"error": "Editor asset not found"})
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        handler._json(422, {"error": str(exc)})
    except OSError:
        handler._json(500, {"error": "Editor storage failed; last published revision is unchanged"})
    return True

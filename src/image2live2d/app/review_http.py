"""HTTP boundary for read-only pre-rig review and explicit approval."""
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .editor import Conflict
from .review import store


def dispatch(handler, method):
    url = urlparse(handler.path)
    if method == 'GET' and url.path in {'/review', '/review.js'}:
        path = Path(__file__).with_name('review.js' if url.path.endswith('.js') else 'review.html')
        handler._send(200, path.read_bytes(), 'text/javascript' if path.suffix == '.js' else 'text/html; charset=utf-8')
        return True
    segments = url.path.strip('/').split('/')
    if segments[:2] != ['api', 'reviews']:
        return False
    try:
        db = store()
        if method == 'GET' and len(segments) == 2:
            handler._json(200, db.list())
            return True
        if len(segments) not in {3, 4}:
            raise ValueError('Unknown review route')
        pid = segments[2]
        action = segments[3] if len(segments) == 4 else ''
        if method == 'POST':
            if handler.headers.get('Origin') != 'http://' + handler.headers.get('Host', ''):
                handler._json(403, {'error': 'Use the review page in this app'})
                return True
            size = int(handler.headers.get('Content-Length', '0'))
            if not 0 < size <= 4096:
                raise ValueError('Invalid approval size')
            data = json.loads(handler.rfile.read(size))
            if action != 'approve' or not isinstance(data, dict):
                raise ValueError('Unknown review action')
            handler._json(202, db.approve(pid, data['content_hash'], data.get('approved')))
        elif method == 'GET' and not action:
            handler._json(200, db.info(pid))
        elif method == 'GET' and action == 'image':
            q = parse_qs(url.query)
            handler._send(200, db.image(pid, q.get('part', ['composite'])[0], q.get('view', ['rgba'])[0]), 'image/png')
        elif method == 'GET' and action == 'download':
            doc = db.info(pid)
            if not doc['last_good']:
                raise Conflict('No successful build yet')
            path = db.path(pid) / 'attempts' / doc['last_good'] / 'model.zip'
            handler.send_response(200)
            handler.send_header('Content-Type', 'application/zip')
            handler.send_header('Content-Disposition', 'attachment; filename="reviewed_vtube.zip"')
            handler.send_header('Content-Length', str(path.stat().st_size))
            handler.end_headers()
            with path.open('rb') as stream:
                import shutil
                shutil.copyfileobj(stream, handler.wfile)
        else:
            raise ValueError('Unknown review route')
    except Conflict as exc:
        handler._json(409, {'error': str(exc)})
    except FileNotFoundError:
        handler._json(404, {'error': 'Review project or asset missing'})
    except (ValueError, KeyError, TypeError) as exc:
        handler._json(422, {'error': str(exc)})
    except OSError:
        handler._json(500, {'error': 'Review storage failed; original snapshot is preserved'})
    return True

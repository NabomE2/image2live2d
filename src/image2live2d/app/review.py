"""Persistent pre-rig review snapshots. CPU builds never mutate approved inputs."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import threading
import uuid
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from .editor import Conflict, atomic_json, digest, identifier, locked, read, sync_dir, sync_tree

POLICY = {"version": 1, "prepare": "legacy_prepare_meshes",
          "description": "기존 자동 준비: missing-part synthesis, limb splitting/role repair, face/leg/accessory order repair"}


def policy():
    from .. import pipeline
    # Pin the entire installed application source: reject resuming under changed preparation code.
    root = Path(pipeline.__file__).parent
    hashes = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*.py'))}
    return {**POLICY, "code_hash": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def store():
    from .server import _web_root
    return ReviewStore(os.environ.get('IMAGE2LIVE2D_REVIEW_DIR', _web_root() / 'review-projects'))


class ReviewStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, pid):
        p = self.root / identifier(pid)
        if not (p / 'project.json').is_file():
            raise FileNotFoundError('Review project not found')
        return p

    def list(self):
        return [read(p) for p in sorted(self.root.glob('*/project.json'))
                if not p.parent.name.startswith('.')]

    def info(self, pid):
        import fcntl
        p = self.path(pid)
        doc = read(p / 'project.json')
        if doc['status'] == 'running':
            with (p / '.build-lock').open('a') as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return doc
                # Display recovery state without overwriting a concurrent request's manifest.
                doc['status'] = 'interrupted'
        return doc

    def create(self, job):
        pid = uuid.uuid4().hex
        temp = self.root / ('.building-' + pid)
        original = temp / 'original'
        (original / 'parts').mkdir(parents=True)
        try:
            (original / 'source').write_bytes(job.source)
            psds = []
            for i, path in enumerate(sorted(job.work.rglob('*.psd'))):
                name = f'psd/{i}.psd'
                (original / 'psd').mkdir(exist_ok=True)
                shutil.copyfile(path, original / name)
                psds.append({'asset': name, 'name': path.name})
            if Path(job.filename).suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.psd'} and not psds:
                raise ValueError('Decomposition PSD missing; cannot publish review snapshot')
            parts = []
            for layer in job.stack.layers:
                item = asdict(layer)
                item['semantic_role'] = layer.semantic_role.value
                item['texture_path'] = 'parts/' + Path(layer.texture_path).name
                if (original / item['texture_path']).exists():
                    raise ValueError('Duplicate part asset name')
                shutil.copyfile(layer.texture_path, original / item['texture_path'])
                parts.append(item)
            snapshot = {'canvas_width': job.stack.canvas_width, 'canvas_height': job.stack.canvas_height,
                        'parts': parts, 'psds': psds, 'filename': job.filename,
                        'decomposition': job.decomposition, 'policy': policy()}
            atomic_json(original / 'snapshot.json', snapshot)
            hashes = {str(p.relative_to(original)): digest(p) for p in original.rglob('*') if p.is_file()}
            content = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
            doc = {'schema': 1, 'id': pid, 'name': job.name, 'status': 'review_required',
                   'content_hash': content, 'hashes': hashes, 'snapshot': snapshot,
                   'approval': None, 'last_good': None, 'error': ''}
            atomic_json(temp / 'project.json', doc)
            (temp / 'attempts').mkdir()
            sync_tree(temp)
            os.replace(temp, self.root / pid)
            sync_dir(self.root)
            return doc
        finally:
            if temp.exists():
                shutil.rmtree(temp)

    def verify(self, pid):
        p = self.path(pid)
        doc = read(p / 'project.json')
        actual = {str(f.relative_to(p / 'original')): digest(f)
                  for f in (p / 'original').rglob('*') if f.is_file()}
        if actual != doc['hashes']:
            raise Conflict('Review snapshot changed; approval refused')
        if doc['snapshot']['policy'] != policy():
            raise Conflict('Preparation code changed; review with the original app version required')
        return doc

    def approve(self, pid, content_hash, approved):
        import fcntl
        if approved is not True:
            raise ValueError('Explicit approval required')
        p = self.path(pid)
        with locked(p):
            doc = self.verify(pid)
            if content_hash != doc['content_hash']:
                raise Conflict('Review content changed')
            if doc['status'] == 'ready':
                return doc
            stream = (p / '.build-lock').open('a')
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                return doc
            try:
                attempt = uuid.uuid4().hex
                doc.update(status='running', error='', attempt=attempt,
                           approval={'content_hash': content_hash, 'policy': doc['snapshot']['policy']})
                atomic_json(p / 'project.json', doc)
                worker = threading.Thread(target=self.build, args=(pid, attempt, stream), daemon=True)
                worker.start()
            except BaseException:
                stream.close()
                raise
            return doc

    def load_approved(self, pid, attempt, destination):
        from ..core.types import Layer, LayerStack
        from ..irr.schema import SemanticRole
        doc = self.verify(pid)
        if (doc['status'] != 'running' or doc.get('attempt') != attempt or
                doc['approval'] != {'content_hash': doc['content_hash'], 'policy': doc['snapshot']['policy']}):
            raise Conflict('No valid approval for this build')
        snap = doc['snapshot']
        shutil.copytree(self.path(pid) / 'original/parts', destination)
        layers = []
        for part in snap['parts']:
            item = dict(part)
            item['texture_path'] = destination / Path(part['texture_path']).name
            item['semantic_role'] = SemanticRole(part['semantic_role'])
            layers.append(Layer(**item))
        return LayerStack(layers, snap['canvas_width'], snap['canvas_height'])

    def build(self, pid, attempt, stream):
        from ..backends.live2d.bundle import bundle_zip, validate_bundle
        from . import server
        p = self.path(pid)
        work = p / 'attempts' / attempt
        try:
            work.mkdir()
            doc = self.verify(pid)
            job = server._Job(id=attempt, name=doc['name'], work=work,
                              mode='manual', filename=doc['snapshot']['filename'], source=(p / 'original/source').read_bytes())
            server._run_pipeline(job, job.source, job.filename, review_context=(self, pid, attempt))
            if job.status != 'done':
                raise ValueError(job.error or 'CPU rigging failed')
            bundle = server.live2d_bundle_dir(job)
            validate_bundle(bundle / 'model.model3.json')
            (work / 'model.zip').write_bytes(bundle_zip(bundle / 'model.model3.json'))
            (work / 'model.inp').write_bytes(job.inp_bytes)
            atomic_json(work / 'rig.json', job.rig.model_dump(mode='json'))
            sync_tree(work)
            self.verify(pid)
            with locked(p):
                doc = read(p / 'project.json')
                doc.update(status='ready', last_good=attempt, error='')
                atomic_json(p / 'project.json', doc)
        except Exception as exc:  # noqa: BLE001 — worker boundary records failures without losing inputs
            with locked(p):
                doc = read(p / 'project.json')
                doc.update(status='failed', error=str(exc))
                atomic_json(p / 'project.json', doc)
        finally:
            stream.close()

    def image(self, pid, selection, view):
        p = self.path(pid)
        doc = read(p / 'project.json')
        snap = doc['snapshot']
        if selection == 'source':
            with Image.open(p / 'original/source') as src:
                im = src.convert('RGBA')
        elif selection == 'composite':
            im = Image.new('RGBA', (snap['canvas_width'], snap['canvas_height']))
            for part in sorted(snap['parts'], key=lambda part: part['draw_order']):
                with Image.open(p / 'original' / part['texture_path']) as src:
                    im.alpha_composite(src.convert('RGBA'))
        else:
            part = next((part for part in snap['parts'] if part['id'] == selection), None)
            if part is None:
                raise ValueError('Unknown part')
            with Image.open(p / 'original' / part['texture_path']) as src:
                im = src.convert('RGBA')
        if view == 'alpha':
            im = im.getchannel('A').convert('RGB')
        elif view == 'rgb':
            im = im.convert('RGB')
        elif view != 'rgba':
            raise ValueError('Unknown view')
        im.thumbnail((1200, 1200))
        out = io.BytesIO()
        im.save(out, format='PNG')
        return out.getvalue()

"""Pre-rig gate tests using production CPU mesh/rig/export and a fake GPU boundary."""
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from image2live2d.app import review, server
from image2live2d.app.editor import Conflict, digest, read
from image2live2d.core import decompose


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv('IMAGE2LIVE2D_JOB_DIR', str(tmp_path / 'jobs'))
    monkeypatch.setenv('IMAGE2LIVE2D_REVIEW_DIR', str(tmp_path / 'reviews'))
    monkeypatch.setenv('IMAGE2LIVE2D_DECOMPOSE_URL', 'http://unused')
    calls = []

    def service(image, target, work_dir, **kwargs):
        from psd_tools import PSDImage
        work_dir.mkdir(parents=True, exist_ok=True)
        PSDImage.new('RGB', (64, 64), color=(200, 180, 160)).save(work_dir / 'source.psd')
        for name, box, color in [('00_face_base', (12, 5, 52, 48), (220, 190, 170, 255)),
                                 ('01_mouth', (24, 30, 40, 33), (60, 20, 20, 255))]:
            img = Image.new('RGBA', (64, 64))
            img.paste(color, box)
            img.save(work_dir / (name + '.png'))
        calls.append('gpu')
        return decompose.from_layer_dir(work_dir)

    monkeypatch.setattr(decompose, 'from_service', service)
    data = io.BytesIO()
    Image.new('RGBA', (64, 64), (200, 180, 160, 255)).save(data, format='PNG')
    return tmp_path, data.getvalue(), calls


def manual(setup):
    root, data, _calls = setup
    work = root / 'source-job'
    work.mkdir(exist_ok=True)
    job = server._Job(id='abcd', name='hero', work=work, source=data, filename='hero.png', mode='manual')
    server._run_pipeline(job, data, 'hero.png')
    assert job.status == 'review_required', job.error
    return job, review.store(), read(review.store().path(job.review_project) / 'project.json')


def finish(db, pid):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        doc = db.info(pid)
        if doc['status'] in {'ready', 'failed'}:
            assert doc['status'] == 'ready', doc.get('error')
            return doc
        time.sleep(.02)
    pytest.fail('CPU build did not finish')


def test_gate_and_recovery_without_gpu(setup, monkeypatch):
    from image2live2d import pipeline
    actual = pipeline.prepare_meshes
    calls = []
    monkeypatch.setattr(pipeline, 'prepare_meshes', lambda stack: calls.append('meshes') or actual(stack))
    _job, db, doc = manual(setup)
    assert calls == [] and _job.rig is None
    assert [s.name for s in _job.steps] == ['Decompose']
    hashes = dict(doc['hashes'])
    server._JOBS.clear()
    reopened = review.ReviewStore(db.root)
    assert reopened.info(doc['id'])['status'] == 'review_required'
    assert doc['snapshot']['psds']
    assert (db.path(doc['id']) / 'original/source').read_bytes() == setup[1]
    monkeypatch.setattr(decompose, 'from_service', lambda *a, **k: pytest.fail('GPU called'))
    reopened.approve(doc['id'], doc['content_hash'], True)
    result = finish(reopened, doc['id'])
    assert calls == ['meshes']
    assert reopened.verify(doc['id'])['hashes'] == hashes
    assert (db.path(doc['id']) / 'attempts' / result['last_good'] / 'model.zip').is_file()
    assert reopened.approve(doc['id'], doc['content_hash'], True)['last_good'] == result['last_good']


def test_auto_same_cpu_outputs(setup):
    _job, db, doc = manual(setup)
    db.approve(doc['id'], doc['content_hash'], True)
    result = finish(db, doc['id'])
    work = setup[0] / 'auto'
    work.mkdir()
    auto = server._Job(id='aaaa', name='hero', work=work, source=setup[1], filename='hero.png')
    server._run_pipeline(auto, auto.source, auto.filename)
    assert auto.status == 'done', auto.error
    approved_ir = read(db.path(doc['id']) / 'attempts' / result['last_good'] / 'rig.json')
    auto_ir = auto.rig.model_dump(mode='json')
    # Asset path roots differ by attempt; all model data must agree.
    for model in (auto_ir, approved_ir):
        for tex in model['textures']:
            tex['path'] = tex['path'].split('/')[-1]
    assert auto_ir == approved_ir
    automatic_bundle = server.live2d_bundle_dir(auto)
    reviewed_bundle = db.path(doc["id"]) / "attempts" / result["last_good"] / "live2d"
    for name in ("model.moc3", "textures/atlas.png"):
        assert digest(automatic_bundle / name) == digest(reviewed_bundle / name)


def test_concurrent_approval(setup, monkeypatch):
    _job, db, doc = manual(setup)
    entered, release = threading.Event(), threading.Event()
    original = db.build
    count = []

    def build(*args):
        count.append(1)
        entered.set()
        release.wait(10)
        original(*args)

    monkeypatch.setattr(db, 'build', build)
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: db.approve(doc['id'], doc['content_hash'], True), range(4)))
    assert entered.wait(2)
    assert len({r['attempt'] for r in results}) == 1
    assert len(count) == 1
    release.set()
    finish(db, doc['id'])


def test_approval_guards_and_tamper(setup):
    _job, db, doc = manual(setup)
    with pytest.raises(ValueError):
        db.approve(doc['id'], doc['content_hash'], False)
    with pytest.raises(Conflict):
        db.approve(doc['id'], 'wrong', True)
    with pytest.raises(Conflict):
        db.load_approved(doc['id'], 'fake', setup[0] / 'bad')
    part = db.path(doc['id']) / 'original' / doc['snapshot']['parts'][0]['texture_path']
    part.write_bytes(b'corrupt')
    with pytest.raises(Conflict):
        db.approve(doc['id'], doc['content_hash'], True)


def test_interrupted_approval_can_retry(setup, monkeypatch):
    _job, db, doc = manual(setup)
    doc.update(status='running', attempt='dead', approval={'content_hash': doc['content_hash'],
                                                        'policy': doc['snapshot']['policy']})
    review.atomic_json(db.path(doc['id']) / 'project.json', doc)
    assert db.info(doc['id'])['status'] == 'interrupted'
    db.approve(doc['id'], doc['content_hash'], True)
    finish(db, doc['id'])


def test_failed_build_preserves_original(setup, monkeypatch):
    from image2live2d import pipeline
    _job, db, doc = manual(setup)
    def fail(stack):
        raise ValueError('injected mesh failure')
    monkeypatch.setattr(pipeline, 'prepare_meshes', fail)
    db.approve(doc['id'], doc['content_hash'], True)
    deadline = time.monotonic() + 5
    while db.info(doc['id'])['status'] == 'running' and time.monotonic() < deadline:
        time.sleep(.02)
    assert db.info(doc['id'])['status'] == 'failed'
    assert db.verify(doc['id'])['hashes'] == doc['hashes']


def test_retention_and_images(setup, monkeypatch):
    _job, db, doc = manual(setup)
    monkeypatch.setenv('IMAGE2LIVE2D_RETENTION_SECONDS', '1')
    server._cleanup_web_jobs()
    for selection in ['composite', doc['snapshot']['parts'][0]['id'], 'source']:
        for view in ['rgba', 'rgb', 'alpha']:
            im = Image.open(io.BytesIO(db.image(doc['id'], selection, view)))
            assert im.size == (64, 64)
    assert db.info(doc['id'])['status'] == 'review_required'


def test_http_browser_review_approval_and_download(setup):
    import shutil
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    playwright = pytest.importorskip('playwright.sync_api')
    chromium = shutil.which('chromium')
    if not chromium:
        pytest.skip('Chromium unavailable')
    _job, _db, doc = manual(setup)
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), server._make_handler())
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{httpd.server_port}'
    try:
        request = urllib.request.Request(base + '/api/reviews/' + doc['id'] + '/approve',
                                         data=b'{}', headers={'Origin': 'http://bad.test'})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request)
        assert exc.value.code == 403
        with playwright.sync_playwright() as engine:
            browser = engine.chromium.launch(executable_path=chromium, args=['--no-sandbox'])
            page = browser.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(base + '/')
            assert page.locator('#generation-mode').input_value() == 'auto'
            page.select_option('#generation-mode', 'manual')
            page.goto(base + '/review?project=' + doc['id'])
            page.wait_for_function("document.querySelector('#status').textContent.includes('검수 대기')")
            page.select_option('#parts', doc['snapshot']['parts'][0]['id'])
            page.select_option('#view', 'alpha')
            page.wait_for_function("document.querySelector('#result').complete && document.querySelector('#result').naturalWidth > 0")
            page.reload()
            page.wait_for_selector('#approve:enabled')
            import os
            if os.environ.get('REVIEW_TEST_SCREENSHOT'):
                page.screenshot(path=os.environ['REVIEW_TEST_SCREENSHOT'], full_page=True)
            page.on('dialog', lambda dialog: dialog.accept())
            page.click('#approve')
            page.wait_for_function("document.querySelector('#status').textContent.includes('모델 생성 완료')")
            with page.expect_download() as download:
                page.click('#download')
            assert download.value.failure() is None
            page.reload()
            page.wait_for_function("!document.querySelector('#download').hidden")
            assert not errors
            browser.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_reopen_in_fresh_process(setup):
    import os
    import subprocess
    import sys
    _job, _db, doc = manual(setup)
    script = "from image2live2d.app.review import store; import sys; d=store().info(sys.argv[1]); assert d['status']=='review_required'; store().verify(sys.argv[1])"
    result = subprocess.run([sys.executable, '-c', script, doc['id']], env=os.environ.copy(),
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_commit_failure_keeps_last_good(setup, monkeypatch):
    _job, db, doc = manual(setup)
    db.approve(doc['id'], doc['content_hash'], True)
    good = finish(db, doc['id'])
    # Simulate a recovered/retryable attempt while retaining the last published build.
    good['status'] = 'failed'
    review.atomic_json(db.path(doc['id']) / 'project.json', good)
    atomic = review.atomic_json
    def fail_publication(path, value):
        if path.name == 'project.json' and value.get('status') == 'ready':
            raise OSError('disk publication failure')
        return atomic(path, value)
    monkeypatch.setattr(review, 'atomic_json', fail_publication)
    db.approve(doc['id'], doc['content_hash'], True)
    deadline = time.monotonic() + 10
    while db.info(doc['id'])['status'] == 'running' and time.monotonic() < deadline:
        time.sleep(.02)
    failed = db.info(doc['id'])
    assert failed['status'] == 'failed'
    assert failed['last_good'] == good['last_good']
    assert (db.path(doc['id']) / 'attempts' / failed['last_good'] / 'model.zip').is_file()
    db.verify(doc['id'])


def test_process_exit_after_approval_recovers(setup):
    import os
    import subprocess
    import sys
    _job, db, doc = manual(setup)
    script = '''
import os, sys, threading
from image2live2d.app.review import store
s = store()
d = s.info(sys.argv[1])
s.build = lambda *args: threading.Event().wait(30)
s.approve(d['id'], d['content_hash'], True)
os._exit(0)
'''
    result = subprocess.run([sys.executable, '-c', script, doc['id']], env=os.environ.copy(),
                            capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == 0, result.stderr
    assert db.info(doc['id'])['status'] == 'interrupted'
    assert db.info(doc['id'])['approval']['content_hash'] == doc['content_hash']
    db.approve(doc['id'], doc['content_hash'], True)
    finish(db, doc['id'])


def test_snapshot_failure_does_not_rig(setup, monkeypatch):
    from image2live2d import pipeline
    monkeypatch.setattr(pipeline, 'prepare_meshes', lambda _: pytest.fail('unapproved mesh'))
    def fail(_self, _job):
        raise OSError('snapshot disk full')
    monkeypatch.setattr(review.ReviewStore, 'create', fail)
    work = setup[0] / 'failed-save'
    work.mkdir()
    job = server._Job(id='aaaa', name='hero', work=work, source=setup[1], filename='hero.png', mode='manual')
    server._run_pipeline(job, job.source, job.filename)
    assert job.status == 'error'
    assert not job.review_project and job.rig is None


def test_manual_job_restart_receipt_and_resume_guard(setup):
    job, _db, doc = manual(setup)
    # Simulate an on-disk web job after loss of the in-memory registry.
    destination = server._web_root() / job.id
    destination.parent.mkdir(parents=True, exist_ok=True)
    job.work.rename(destination)
    (destination / 'upload').write_bytes(setup[1])
    server._JOBS.pop(job.id, None)
    recovered = server.get_job(job.id)
    assert recovered.status == 'review_required'
    assert recovered.mode == 'manual' and recovered.review_project == doc['id']
    with pytest.raises(ValueError, match='review project'):
        server.resume_job(job.id)


def test_http_upload_gate_and_approval(setup):
    import json
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), server._make_handler())
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{httpd.server_port}'
    def request(path, data=None, origin=base):
        req = urllib.request.Request(base + path, data=data, headers={'Origin': origin})
        with urllib.request.urlopen(req) as response:
            return json.load(response)
    try:
        created = request('/api/jobs?name=hero.png&mode=manual', setup[1])
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = request('/api/jobs/' + created['job_id'])
            if job['status'] != 'running':
                break
            time.sleep(.02)
        assert job['status'] == 'review_required'
        pid = job['review_project']
        doc = request('/api/reviews/' + pid)
        assert any(p['id'] == pid for p in request('/api/reviews'))
        payload = json.dumps({'content_hash': doc['content_hash'], 'approved': True}).encode()
        with pytest.raises(urllib.error.HTTPError) as exc:
            request('/api/reviews/' + pid + '/approve', payload, 'http://bad.test')
        assert exc.value.code == 403
        request('/api/reviews/' + pid + '/approve', payload)
        finish(review.store(), pid)
        with urllib.request.urlopen(base + '/api/reviews/' + pid + '/download') as response:
            assert response.read(2) == b'PK'
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_expired_job_cleanup_preserves_review(setup):
    import os
    _job, db, doc = manual(setup)
    old = server._web_root() / 'deadbeef'
    old.mkdir(parents=True)
    review.atomic_json(old / 'job.json', {'status': 'done'})
    os.utime(old / 'job.json', (1, 1))
    os.utime(db.path(doc['id']) / 'project.json', (1, 1))
    server._cleanup_web_jobs()
    assert not old.exists()
    assert db.verify(doc['id'])['status'] == 'review_required'

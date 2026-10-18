"""UI/API checkpoint integration without a GPU."""
import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from image2live2d.app import server


def test_web_restart_restores_receipt_and_resume_invalidates_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE2LIVE2D_JOB_DIR", str(tmp_path))
    monkeypatch.setattr(server, "_JOBS", {})
    jid = "abc123"
    work = tmp_path / jid
    work.mkdir()
    (work / "upload").write_bytes(b"source")
    (work / "job.json").write_text(json.dumps({"name": "hero", "filename": "hero.png",
        "updated": time.time(), "status": "error", "decomposition": {"job_id": "remote"}}))
    (work / "live2d").mkdir()
    (work / "live2d/stale.moc3").write_bytes(b"old model")
    job = server.get_job(jid)
    assert job.source == b"source" and job.decomposition["job_id"] == "remote"
    starts = []
    class Thread:
        def __init__(self, **kwargs):
            starts.append(kwargs)
        def start(self):
            pass
    monkeypatch.setattr(server.threading, "Thread", Thread)
    assert server.resume_job(jid, "depth") == jid
    assert job.status == "running" and job.restart_stage == "depth"
    assert not (work / "live2d").exists()
    with pytest.raises(ValueError, match="already running"):
        server.resume_job(jid)
    assert len(starts) == 1


def test_web_cleanup_retains_active_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE2LIVE2D_JOB_DIR", str(tmp_path))
    monkeypatch.setenv("IMAGE2LIVE2D_RETENTION_SECONDS", "1")
    monkeypatch.setattr(server, "_JOBS", {})
    import os
    for jid, status in (("a", "running"), ("b", "error")):
        work = tmp_path / jid
        work.mkdir()
        record = work / "job.json"
        record.write_text('{}')
        os.utime(record, (time.time() - 10, time.time() - 10))
        server._JOBS[jid] = server._Job(id=jid, name=jid, status=status)
    server._cleanup_web_jobs()
    assert (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_depth_retry_ui_uses_existing_job_and_renders_stage_errors():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    page = (Path(__file__).parents[1] / "src/image2live2d/app/index.html").read_text()
    code = page[page.index('async function poll(){'):page.index("$('#resume-job').addEventListener")]
    script = '''
const assert=require('assert');
let jobId='abc',pollTimer=1;
const controls={};
const $=id=>controls[id]||(controls[id]={style:{},disabled:false});
const clearInterval=()=>{},setInterval=()=>2;
const renderSteps=()=>{},stopTicker=()=>{},onDone=()=>{},escapeHtml=x=>x;
let calls=[];
let status={status:'error',steps:[],error:'Depth timeout',decomposition:{job_id:'remote',stages:{
 layerdiff:{status:'done',reused:true},depth:{status:'error',error:'timeout'},psd:{status:'pending'}}}};
const fetch=async(url,options)=>{
 calls.push([url,options]);
 return {ok:true,json:async()=>options?{job_id:'abc'}:status};
};
''' + code + '''
(async()=>{
 await poll();
 assert(!$('#resume-job').hidden && !$('#retry-depth').hidden);
 assert($('#depth-stages').textContent.includes('layerdiff: done (reused)'));
 assert($('#depth-stages').textContent.includes('depth: error'));
 await resumeConversion('depth');
 assert(calls.some(([url,options])=>url==='/api/jobs/abc/resume?stage=depth' && options.method==='POST'));
 assert(!calls.some(([url])=>url==='/api/jobs'));
 status.status='running'; await poll();
 assert($('#retry-depth').hidden);
})().catch(e=>{console.error(e);process.exit(1)});
'''
    result = subprocess.run([node, "-e", script], text=True, capture_output=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr


def test_resume_rejects_worker_still_finalizing(monkeypatch):
    job = server._Job(id="abc", name="test", status="error", worker_active=True,
                      decomposition={"job_id": "remote"})
    monkeypatch.setattr(server, "_JOBS", {"abc": job})
    with pytest.raises(ValueError, match="already running"):
        server.resume_job("abc")

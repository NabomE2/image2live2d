"""Exercise download UI states in Node; browser rendering remains separate."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_download_ui_never_enables_for_failed_export():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable; download UI state transitions untested")
    page = (Path(__file__).parents[1] / "src/image2live2d/app/index.html").read_text()
    code = page[page.index("async function prepareVtube("):page.index(
        "// ====================  In-browser live runtime")]
    script = '''
const assert=require('assert');
let jobId='current', handler, downloaded;
const controls={
  '#dl-vtube':{disabled:true,addEventListener:(event,fn)=>handler=fn},
  '#vtube-status':{textContent:''}
};
const $=selector=>controls[selector];
const document={createElement:()=>({click(){downloaded=this.download;},remove(){}}),
                body:{appendChild(){}}};
const URL={createObjectURL:()=> 'blob:fixture',revokeObjectURL(){}};
const setTimeout=fn=>fn();
let fetch;
''' + code + '''
(async()=>{
  fetch=async()=>({ok:false,json:async()=>({error:'Missing moc3'})});
  await prepareVtube('current');
  assert(controls['#dl-vtube'].disabled);
  assert(controls['#vtube-status'].textContent.includes('Missing moc3'));
  fetch=async()=>({ok:true,json:async()=>({ready:true})});
  await prepareVtube('current');
  assert(!controls['#dl-vtube'].disabled);
  assert(controls['#vtube-status'].textContent.includes('미검증'));
  fetch=async()=>({ok:false,json:async()=>({error:'Missing texture'})});
  await handler();
  assert(controls['#dl-vtube'].disabled);
  assert(controls['#vtube-status'].textContent.includes('Missing texture'));
  assert.strictEqual(downloaded,undefined);
  fetch=async()=>({ok:true,blob:async()=>Buffer.from('zip'),
    headers:{get:()=> 'attachment; filename="hero_vtube.zip"'}});
  await handler();
  assert.strictEqual(downloaded,'hero_vtube.zip');
  assert(controls['#vtube-status'].textContent.includes('Ctrl+J'));
})().catch(error=>{console.error(error);process.exit(1);});
'''
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr

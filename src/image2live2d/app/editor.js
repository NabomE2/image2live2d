'use strict';
const $=id=>document.getElementById(id);
let project=null,part=null,draft=null,original=null,edited=null,busy=false;
let zoom=1,pan={x:0,y:0},gesture=null,points=[];
const base='/api/editor/projects';
async function request(path,body){
  const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const doc=await r.json(); if(!r.ok)throw new Error(doc.error||'요청 실패'); return doc;
}
function message(text){$('status').textContent=text;}
function controls(){
  for(const id of ['projects','parts','revisions','restore','download','resume-draft','reset'])$(id).disabled=busy;
  $('apply').disabled=busy||!draft;
}
async function action(fn){if(busy)return;busy=true;controls();$('error').textContent='';
  try{await fn();}catch(e){$('error').textContent=e.message;message('요청 실패. 마지막 저장 버전은 유지됩니다. 충돌이면 현재 버전을 다시 여세요.');}
  finally{busy=false;controls();}}
function options(select,items,value,label){select.replaceChildren();for(const item of items){const option=document.createElement('option');option.value=value(item);option.textContent=label(item);select.appendChild(option);}}
async function list(){const items=await request(base);options($('projects'),[{id:'',name:'프로젝트 선택'},...items],p=>p.id,p=>p.name+' '+p.id.slice(0,8));}
async function openProject(id){
  if(!id){$('workspace').hidden=true;return;}
  project=await request(base+'/'+id);$('workspace').hidden=false;
  options($('parts'),project.parts,p=>p.id,p=>p.id+(p.editable?' · 입 편집':' · 보기 전용'));
  options($('revisions'),project.revisions.slice().reverse(),r=>r.id,r=>r.kind+' '+r.id.slice(0,8));
  $('revisions').value=project.current;
  options($('drafts'),[{id:'',part_id:'초안 선택'},...(project.drafts||[])],d=>d.id,d=>d.part_id+' '+d.id.slice(0,8));
  $('parts').value=project.parts.find(p=>p.editable)?.id||project.parts[0]?.id;
  $('project-info').textContent='현재 버전 '+project.current.slice(0,8)+' · 원본 및 .moc3 일치 검사 통과 · GPU 재추론 없음';
  history.replaceState(null,'','/editor?project='+id);
  await loadPart();
}
async function image(url){const r=await fetch(url);if(!r.ok)throw new Error('이미지를 불러오지 못했습니다');const blob=await r.blob();return createImageBitmap(blob);}
function asset(revision,did){const q=new URLSearchParams({part:part.id});if(did)q.set('draft',did);else q.set('revision',revision);return base+'/'+project.id+'/asset?'+q;}
async function loadPart(resume=null){
  draft=resume;part=project.parts.find(p=>p.id===$('parts').value);
  if(!part)return;
  const pair=await Promise.all([image(asset('original')),image(asset($('revisions').value,draft?.id))]);
  original?.close();edited?.close();[original,edited]=pair;fit();
  message(draft?'저장된 초안을 열었습니다. 비교 후 승인하세요.':part.editable?'입 alpha를 편집할 수 있습니다.':'이 파츠는 보기 전용입니다.');controls();
}
function fit(){if(!original)return;const c=$('edited');zoom=Math.min(c.clientWidth/original.width,c.clientHeight/original.height)*.9;
 pan={x:(c.clientWidth-original.width*zoom)/2,y:(c.clientHeight-original.height*zoom)/2};draw();}
function draw(){
 for(const [id,img] of [['original',original],['edited',edited]]){
  if(!img)continue;const c=$(id);c.className=$('background').value;c.width=c.clientWidth;c.height=c.clientHeight;const ctx=c.getContext('2d');
  ctx.imageSmoothingEnabled=zoom<3;
  let source=img;
  if($('alpha').checked){const t=document.createElement('canvas');t.width=img.width;t.height=img.height;const tc=t.getContext('2d');tc.drawImage(img,0,0);const rgba=tc.getImageData(0,0,t.width,t.height);for(let i=0;i<rgba.data.length;i+=4){rgba.data[i]=rgba.data[i+1]=rgba.data[i+2]=rgba.data[i+3];rgba.data[i+3]=255;}tc.putImageData(rgba,0,0);source=t;}
  ctx.drawImage(source,pan.x,pan.y,img.width*zoom,img.height*zoom);
 }
}
function position(e){const r=$('edited').getBoundingClientRect();return{x:(e.clientX-r.left-pan.x)/zoom,y:(e.clientY-r.top-pan.y)/zoom};}
function dab(p){return{x:Math.max(0,Math.min(part.width,p.x)),y:Math.max(0,Math.min(part.height,p.y)),radius:Number($('radius').value),hardness:Number($('hardness').value),mode:$('tool').value};}
$('edited').addEventListener('pointerdown',e=>{
 if(busy||!part)return;
 const move=$('tool').value==='pan'||e.shiftKey;
 if(!move&&(!part.editable||$('revisions').value!==project.current)){message('현재 버전의 입 파츠를 선택해야 편집할 수 있습니다.');return;}
 gesture={move,x:e.clientX,y:e.clientY,last:position(e)};points=move?[]:[dab(gesture.last)];$('edited').setPointerCapture(e.pointerId);
});
$('edited').addEventListener('pointermove',e=>{
 if(!gesture)return;
 if(gesture.move){pan.x+=e.clientX-gesture.x;pan.y+=e.clientY-gesture.y;gesture.x=e.clientX;gesture.y=e.clientY;draw();return;}
 const p=position(e),prev=gesture.last,dist=Math.hypot(p.x-prev.x,p.y-prev.y),step=Math.max(.5,Number($('radius').value)/3),n=Math.ceil(dist/step);
 if(points.length+n>4096){message('긴 스트로크는 나누어 그려 주세요.');return;}
 for(let i=1;i<=n;i++)points.push(dab({x:prev.x+(p.x-prev.x)*i/n,y:prev.y+(p.y-prev.y)*i/n}));gesture.last=p;
 // Overlay the brush path until the server returns authoritative alpha pixels.
 draw();const ctx=$('edited').getContext('2d');ctx.strokeStyle='#ec3c75';ctx.lineWidth=1;ctx.beginPath();ctx.arc(p.x*zoom+pan.x,p.y*zoom+pan.y,Number($('radius').value)*zoom,0,Math.PI*2);ctx.stroke();
});
async function endStroke(){if(!gesture)return;const moving=gesture.move;gesture=null;if(moving)return;const strokes=points;points=[];
 await action(async()=>{
  if(!draft)draft=await request(base+'/'+project.id+'/drafts',{base:project.current,part_id:part.id});
  for(let i=0;i<strokes.length;i+=256)draft=await request(base+'/'+project.id+'/strokes',{draft:draft.id,sequence:draft.sequence,strokes:strokes.slice(i,i+256)});
  const fresh=await image(asset(null,draft.id));edited.close();edited=fresh;draw();message('초안 저장 완료. 왼쪽 원본과 비교한 뒤 승인하면 새 버전으로 저장됩니다.');
 });}
$('edited').addEventListener('pointerup',endStroke);
$('edited').addEventListener('pointercancel',()=>{gesture=null;points=[];draw();});
$('edited').addEventListener('wheel',e=>{e.preventDefault();if(!edited)return;const p=position(e);const next=Math.max(.05,Math.min(32,zoom*Math.exp(-e.deltaY*.002)));pan.x+=p.x*(zoom-next);pan.y+=p.y*(zoom-next);zoom=next;draw();},{passive:false});
$('projects').onchange=()=>action(()=>openProject($('projects').value));
$('reload').onclick=()=>action(list);
$('parts').onchange=()=>action(()=>loadPart());
$('revisions').onchange=()=>action(()=>loadPart());
$('background').onchange=draw;$('alpha').onchange=draw;$('fit').onclick=fit;
$('radius').oninput=()=>{$('radius-value').textContent=$('radius').value;};
$('apply').onclick=()=>action(async()=>{
 if(!draft||!confirm('표시 중인 초안으로 새 버전을 저장할까요? 원본은 유지됩니다.'))return;
 await request(base+'/'+project.id+'/apply',{draft:draft.id,sequence:draft.sequence,approved:true});
 await openProject(project.id);message('새 버전 저장 완료. .moc3는 그대로 재사용했습니다.');
});
$('restore').onclick=()=>action(async()=>{
 if(!confirm('선택한 버전으로 복원 버전을 만들까요? 기존 이력은 유지됩니다.'))return;
 await request(base+'/'+project.id+'/restore',{target:$('revisions').value,base:project.current,approved:true});
 await openProject(project.id);message('복원 버전을 저장했습니다.');
});
$('reset').onclick=()=>action(()=>openProject(project.id));
$('resume-draft').onclick=()=>action(async()=>{
 project=await request(base+'/'+project.id);const d=(project.drafts||[]).find(x=>x.id===$('drafts').value);if(!d)throw new Error('초안을 선택하세요');
 if(d.base!==project.current)throw new Error('이전 버전의 초안입니다. 현재 버전에서 새 초안을 만드세요.');
 $('parts').value=d.part_id;$('revisions').value=d.base;await loadPart(d);
});
$('download').onclick=()=>{const a=document.createElement('a');a.href=base+'/'+project.id+'/download?revision='+$('revisions').value;a.download='edited_vtube.zip';a.click();};
window.addEventListener('resize',draw);
action(async()=>{await list();const id=new URLSearchParams(location.search).get('project');if(id){$('projects').value=id;await openProject(id);}});

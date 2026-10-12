/* Preset selection, normalized ROI editor and workstation statistics. */
class WorkstationUI {
  constructor({api,flash,onChange,onEditorChange,getJob,isBusy}){
    Object.assign(this,{api,flash,onChange,onEditorChange,getJob,isBusy});
    this.profiles=[];this.available=false;this.scenes={};this.jobs={};this.reports={};this.request=0;this.polling=false;
    this.dialog=document.querySelector('#station-setup-dialog');this.canvas=document.querySelector('#station-canvas');this.points=[];this.draft=null;this.image=null;this.readOnly=false;
    this.q=id=>document.getElementById(id);
    for(const kind of ['monitor','file']){
      const prefix=kind==='monitor'?'m':'f';
      this.q(prefix+'-profile').onchange=()=>{this.updateTask(kind);onChange();};
      this.q(prefix+'-station-setup').onclick=()=>this.setup(kind);
      this.q(prefix+'-station-export').onclick=()=>this.export(kind);
      this.q(prefix+'-station-locate').onclick=()=>this.setup(kind,true);
    }
    this.q('profile-editor').onchange=()=>onEditorChange();
    this.q('profile-copy').onclick=()=>this.copyProfile();
    this.q('profile-default').onclick=()=>this.makeDefault();
    this.q('profile-delete').onclick=()=>this.deleteProfile();
    this.q('station-close').onclick=()=>this.dialog.close();
    this.q('station-add').onclick=()=>this.addStation();
    this.q('station-undo').onclick=()=>{this.points.pop();this.draw();};
    this.q('station-clear').onclick=()=>{this.points=[];this.draw();};
    this.q('station-save').onclick=()=>this.saveScene();
    this.canvas.onclick=event=>{
      if(this.readOnly||!this.image)return;
      if(this.points.length>=20)return flash('单个工位最多20个点','warn');
      const rect=this.canvas.getBoundingClientRect();this.points.push([Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))]);this.draw();
    };
    this.timer=setInterval(()=>this.poll(),1500);
  }
  escape(value){return String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
  async init(){
    try{
      const response=await this.api('/api/profiles');
      if(!Array.isArray(response.profiles)||!response.profiles.length)return;
      this.available=true;this.profiles=response.profiles;
      for(const id of ['m-profile','f-profile','profile-editor']){
        const select=this.q(id),previous=select.value;select.replaceChildren();
        for(const profile of this.profiles)select.append(new Option(profile.name,profile.id));
        select.value=this.initialized&&this.profiles.some(p=>p.id===previous)?previous:response.default_id;
      }
      this.initialized=true;for(const kind of ['monitor','file'])this.updateTask(kind);
      await this.onEditorChange();this.onChange();
    }catch(e){this.flash('行为预设暂不可用：'+e.message,'warn');}
  }
  id(kind){return this.q(kind==='monitor'?'m-profile':'f-profile').value||'current';}
  mode(kind){return this.profiles.find(p=>p.id===this.id(kind))?.mode||'behavior';}
  scope(kind){return kind==='monitor'?{source:this.q('m-src').value.trim()}:{job_id:this.getJob('file')};}
  scene(kind){const scope=this.scope(kind);return this.scenes[kind]?.key===JSON.stringify(scope)?this.scenes[kind].scene:null;}
  task(kind){return {profile_id:this.id(kind),...(this.mode(kind)==='workstation'?{station_scene:this.scene(kind)}:{})};}
  ready(kind){return this.mode(kind)!=='workstation'||!!this.scene(kind);}
  updateTask(kind){
    const prefix=kind==='monitor'?'m':'f',station=this.mode(kind)==='workstation';
    this.q(prefix+'-station-setup').hidden=!station;
    this.q(prefix+'-profile-note').textContent=station?(this.scene(kind)?`已配置 ${this.scene(kind).stations.length} 个工位 · 任务开始后使用独立版本`:'请先绘制工位区域，再开始人员统计'):'本任务使用所选行为预设，已开始的任务不受编辑影响';
  }
  attach(kind,jid,profile=null){
    this.jobs[kind]={id:jid,profile:profile||this.profiles.find(p=>p.id===this.id(kind))};
    const active=this.jobs[kind].profile?.mode==='workstation';this.q((kind==='monitor'?'m':'f')+'-station-panel').hidden=!active;
    if(active)this.poll();
  }
  clear(kind){delete this.jobs[kind];delete this.reports[kind];this.q((kind==='monitor'?'m':'f')+'-station-panel').hidden=true;}
  async poll(){
    if(this.polling||document.hidden)return;this.polling=true;
    try{
      for(const [kind,job] of Object.entries(this.jobs)){
        if(job.profile?.mode!=='workstation')continue;
        try{const report=await this.api('/api/stations/'+encodeURIComponent(job.id)+'?compact=true');if(this.jobs[kind]?.id===job.id)this.render(kind,report);}
        catch(e){this.q((kind==='monitor'?'m':'f')+'-station-status').textContent='统计暂不可用：'+e.message;}
      }
    }finally{this.polling=false;}
  }
  duration(value){const seconds=Math.max(0,Number(value)||0);return seconds>=3600?`${Math.floor(seconds/3600)}时${Math.floor(seconds%3600/60)}分${Math.floor(seconds%60)}秒`:seconds>=60?`${Math.floor(seconds/60)}分${Math.floor(seconds%60)}秒`:`${seconds.toFixed(1)}秒`;}
  qaStatistics(facts){
    if(!facts)return '';
    const names={at_station:'在岗',away:'离岗',unknown:'未知'};
    return `<details class="note" open><summary>本次回答引用的程序统计${facts.use_mock?' · 演示数据':''}</summary><div class="station-table-wrap"><table class="station-table"><thead><tr><th>人员</th><th>当前状态</th><th>累计在岗</th><th>累计离岗</th><th>未知</th><th>离岗次数</th><th>最长离岗</th></tr></thead><tbody>${(facts.people||[]).map(p=>`<tr><td>${this.escape(p.name||p.person_id)} · ${this.escape(p.person_id)}</td><td>${names[p.state]||'未知'}</td><td>${Number(p.at_station_sec).toFixed(1)}秒</td><td>${Number(p.away_sec).toFixed(1)}秒</td><td>${Number(p.unknown_sec).toFixed(1)}秒</td><td>${this.escape(p.away_count)}</td><td>${Number(p.longest_away_sec).toFixed(1)}秒</td></tr>`).join('')}</tbody></table></div><span class="muted">${this.escape(facts.source)}</span></details>`;
  }
  render(kind,report){
    const prefix=kind==='monitor'?'m':'f';this.reports[kind]=report;
    this.q(prefix+'-station-panel').hidden=false;
    this.q(prefix+'-station-status').textContent=`${report.use_mock?'演示数据 · ':''}${report.model_status||'人员观测与大模型复核'}${report.model_error?' · '+report.model_error:''} · 统计到 ${this.duration(report.time_sec)}`;
    this.q(prefix+'-station-counts').innerHTML=(report.stations||[]).map(s=>`<div class="station-metric"><span>${this.escape(s.name)}</span><strong>${s.observed_count==null?'不可观察':this.escape(s.observed_count)+'<small> 人</small>'}</strong><span>观测驻留 ${this.escape(s.resident_count)} · 预期 ${this.escape(s.expected_people)}${s.ai_check?.visible?` · AI 核对 ${this.escape(s.ai_check.count)}（${Number(s.ai_check.time_sec).toFixed(1)}s）`:''}</span></div>`).join('');
    const states={at_station:'在岗',away:'离岗',unknown:'未知'},review={confirmed:'模型已确认',uncertain:'模型待复核',pending:'等待模型复核'};
    this.q(prefix+'-station-people').innerHTML=(report.people||[]).map((p,i)=>`<tr><td><strong>${this.escape(p.name||p.person_id)}</strong><small>${this.escape(p.person_id)}${p.identity_status==='pending'?' · 待关联':''}</small></td><td><span class="station-state ${p.state==='away'?'away':p.state==='at_station'?'present':'unknown'}">${states[p.state]||'未知'}</span><small>${this.escape(review[p.review_status]||'待复核')}</small></td><td>${this.duration(p.current_duration_sec)}</td><td>${this.duration(p.at_station_sec)}</td><td>${this.duration(p.away_sec)}</td><td>${this.duration(p.unknown_sec)}</td><td>${this.escape(p.away_count)}</td><td>${this.duration(p.longest_away_sec)}</td><td><button class="btn ghost sm" data-person="${i}">关联 / 命名</button></td></tr>`).join('')||'<tr><td colspan="9" class="muted">尚未观测到可跟踪人员，不会推断某位员工已离岗。</td></tr>';
    this.q(prefix+'-station-people').querySelectorAll('[data-person]').forEach(button=>button.onclick=()=>this.editPerson(kind,report.people[Number(button.dataset.person)]));
    const timeline=[...(report.timeline||[]),...(report.station_timeline||[]).map(s=>({...s,regional:true,start:s.since}))].sort((a,b)=>b.start-a.start).slice(0,80);
    this.q(prefix+'-station-timeline').innerHTML=timeline.map((span,i)=>{
      const station=report.stations.find(s=>s.id===span.station_id),label=span.regional?`${station?.name||span.station_id} · ${({empty:'人数不足',over:'人数超限',normal:'人数正常'})[span.category]} ${span.count}人`:`${span.person_id} · ${states[span.state]||'未知'}`;
      const css=span.regional?(span.category==='normal'?'at_station':'away'):span.state;
      return `<button class="station-span ${css}" data-span="${i}"><strong>${this.escape(label)}</strong><span>${Number(span.start).toFixed(1)}s — ${span.end==null?'进行中':Number(span.end).toFixed(1)+'s'}</span><span>${this.duration((span.end??report.time_sec)-span.start)} · 查看证据</span></button>`;
    }).join('')||'<span class="muted">时间段将在人员进入、离开或观察中断后显示。</span>';
    this.q(prefix+'-station-timeline').querySelectorAll('[data-span]').forEach(button=>button.onclick=()=>{const span=timeline[Number(button.dataset.span)],bid=span.regional?(span.category==='empty'?'station_empty':'station_over_capacity'):'station_away';openEvidence(this.jobs[kind].id,(span.regional?'station_interval_':'interval_')+span.interval_id,bid,report.analysis_revision);});
  }
  async setup(kind,readOnly=false){
    const scope=this.scope(kind);
    if(kind==='file'&&!scope.job_id&&this.jobs[kind]?.id)scope.job_id=this.jobs[kind].id;
    if(!scope.source&&!scope.job_id)return this.flash(kind==='file'?'请先上传视频':'请先填写摄像头地址','warn');
    this.kind=kind;this.readOnly=readOnly;this.scopeDraft=scope;const request=++this.request;
    this.q('station-setup-status').textContent='正在读取画面和工位配置…';this.q('station-save').disabled=true;
    this.points=[];this.image=null;this.q('station-canvas').style.visibility='hidden';
    this.q('station-edit-controls').hidden=readOnly;this.q('station-save').hidden=readOnly;
    if(!this.dialog.open)this.dialog.showModal();
    try{
      const [response,loaded,profile]=await Promise.all([
        fetch('/api/station-scenes/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(scope)}),
        this.api('/api/station-scenes/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(scope)}),
        this.api('/api/profiles/'+encodeURIComponent(this.id(kind)))
      ]);
      if(!response.ok){const error=await response.json();throw Error(error.detail||'读取画面失败');}
      const blob=await response.blob(),url=URL.createObjectURL(blob),image=new Image();
      try{image.src=url;await image.decode();}finally{URL.revokeObjectURL(url);}
      if(request!==this.request||!this.dialog.open)return;
      this.image=image;this.canvas.width=image.naturalWidth;this.canvas.height=image.naturalHeight;this.canvas.style.visibility='visible';
      const existing=loaded.scene||this.reports[kind]?.scene;
      this.draft=existing?JSON.parse(JSON.stringify(existing)):{stations:[],policy:profile.station_policy||{},start_clock:''};
      for(const [field,key] of [['station-away','away_sec'],['station-empty','empty_sec'],['station-confirm','confirm_sec']])this.q(field).value=this.draft.policy[key]??({away_sec:60,empty_sec:60,confirm_sec:3}[key]);
      this.q('station-work-hours').value=JSON.stringify(this.draft.policy.work_hours||[]);this.q('station-break-hours').value=JSON.stringify(this.draft.policy.break_hours||[]);
      this.q('station-start-clock').value=this.draft.start_clock||'';
      this.q('station-setup-status').textContent=readOnly?'绿色框标注最近观测的匿名人员，画面与统计时间可能存在采样间隔。':'沿座椅或作业区域点击至少3个点，再点“添加工位”。工位区域需包含人员身体中心。';
      this.draw();this.drawList();this.q('station-save').disabled=false;
    }catch(e){this.q('station-setup-status').textContent='读取失败：'+e.message;}
  }
  draw(){
    if(!this.image)return;const context=this.canvas.getContext('2d'),w=this.canvas.width,h=this.canvas.height;
    context.drawImage(this.image,0,0,w,h);
    const polygon=(points,color,label)=>{if(!points.length)return;context.beginPath();points.forEach((p,i)=>i?context.lineTo(p[0]*w,p[1]*h):context.moveTo(p[0]*w,p[1]*h));if(points.length>=3)context.closePath();context.strokeStyle=color;context.lineWidth=3;context.stroke();context.fillStyle=color+'22';context.fill();context.fillStyle=color;context.font='16px Microsoft YaHei';if(label)context.fillText(label,points[0][0]*w+5,points[0][1]*h+20);};
    for(const station of this.draft?.stations||[])polygon(station.polygon,'#4b73a3',station.name);
    polygon(this.points,'#b63832','绘制中');
    if(this.readOnly)for(const p of this.reports[this.kind]?.people||[]){if(!p.visible||!p.bbox)continue;const [x1,y1,x2,y2]=p.bbox;context.strokeStyle='#157347';context.strokeRect(x1*w,y1*h,(x2-x1)*w,(y2-y1)*h);context.fillStyle='#157347';context.fillText(p.name||p.person_id,x1*w,Math.max(18,y1*h));}
  }
  addStation(){
    if(this.points.length<3)return this.flash('请在画面上点击至少3个点','warn');
    if(this.draft.stations.length>=20)return this.flash('最多配置20个工位','warn');
    const expected=Number(this.q('station-expected').value),name=this.q('station-name').value.trim();
    if(!name||!Number.isInteger(expected)||expected<0||expected>30)return this.flash('填写工位名称和0至30的预期人数','warn');
    const id='s'+Date.now().toString(36)+Math.random().toString(36).slice(2,5);
    this.draft.stations.push({id,name,expected_people:expected,polygon:this.points.slice()});this.points=[];this.q('station-name').value=`${this.draft.stations.length+1}号工位`;this.draw();this.drawList();
  }
  drawList(){
    this.q('station-region-list').innerHTML=(this.draft?.stations||[]).map((s,i)=>`<span class="station-region-tag">${this.escape(s.name)} · ${s.expected_people}人${this.readOnly?'':` <button type="button" class="btn ghost sm" data-remove="${i}" aria-label="删除${this.escape(s.name)}">移除</button>`}</span>`).join('');
    this.q('station-region-list').querySelectorAll('[data-remove]').forEach(button=>button.onclick=()=>{this.draft.stations.splice(Number(button.dataset.remove),1);this.draw();this.drawList();});
  }
  async saveScene(){
    if(!this.draft?.stations.length)return this.flash('请至少添加一个工位','warn');
    try{
      this.draft.policy={...this.draft.policy,away_sec:Number(this.q('station-away').value),empty_sec:Number(this.q('station-empty').value),confirm_sec:Number(this.q('station-confirm').value),work_hours:JSON.parse(this.q('station-work-hours').value),break_hours:JSON.parse(this.q('station-break-hours').value)};
      this.draft.start_clock=this.q('station-start-clock').value;
      this.q('station-save').disabled=true;
      const result=await this.api('/api/station-scenes/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...this.scopeDraft,scene:this.draft})});
      this.scenes[this.kind]={key:JSON.stringify(this.scopeDraft),scene:result.scene};this.updateTask(this.kind);this.onChange();this.dialog.close();this.flash('工位区域和规则已保存','ok');
    }catch(e){this.q('station-setup-status').textContent='保存失败：'+e.message;}
    finally{this.q('station-save').disabled=false;}
  }
  async editPerson(kind,person){
    const report=this.reports[kind],dialog=this.q('station-person-dialog');this.personDraft={kind,person,report};
    this.q('station-person-title').textContent='关联人员 · '+person.person_id;this.q('station-person-name').value=person.name||person.person_id;
    this.q('station-person-home').replaceChildren(new Option('保留现有关联',''));
    for(const s of report.stations)this.q('station-person-home').append(new Option(s.name,s.id));
    this.q('station-person-merge').replaceChildren(new Option('不合并，只修改名称或工位',''));
    for(const p of report.people.filter(p=>p.person_id!==person.person_id))this.q('station-person-merge').append(new Option(`${p.name} · ${p.person_id}`,p.person_id));
    this.q('station-person-status').textContent='仅对不重叠的人员观测进行人工关联，不自动识别实名。';dialog.showModal();
    this.q('station-person-close').onclick=()=>dialog.close();
    this.q('station-person-save').onclick=async()=>{
      const merge=this.q('station-person-merge').value;
      const body={analysis_revision:report.analysis_revision,person_id:person.person_id,...(merge?{merge_into:merge}:{name:this.q('station-person-name').value,station_id:this.q('station-person-home').value||null})};
      this.q('station-person-save').disabled=true;
      try{await this.api(`/api/stations/${encodeURIComponent(this.jobs[kind].id)}/person`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});dialog.close();this.poll();}
      catch(e){this.q('station-person-status').textContent='保存失败：'+e.message;}
      finally{this.q('station-person-save').disabled=false;}
    };
  }
  download(value,name){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json;charset=utf-8'}));const link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  async export(kind){try{const job=this.jobs[kind];if(!job)return;const report=await this.api('/api/stations/'+encodeURIComponent(job.id));this.download(report,job.id+'-workstations.json');}catch(e){this.flash('导出失败：'+e.message,'err');}}
  async editorLoad(){const id=this.q('profile-editor').value||'current';return this.api(`/api/profiles/${encodeURIComponent(id)}/raw`);}
  async editorSave(text){const id=this.q('profile-editor').value||'current';const result=await this.api(`/api/profiles/${encodeURIComponent(id)}/raw`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});await this.init();return result;}
  async editorPrompt(){const id=this.q('profile-editor').value||'current';return this.api(`/api/profiles/${encodeURIComponent(id)}/prompt`);}
  async copyProfile(){
    try{const selected=this.q('profile-editor').value;const doc=await this.api('/api/profiles/'+encodeURIComponent(selected));const id='custom-'+Date.now().toString(36);const name=this.q('profile-copy-name').value.trim()||doc.name+'副本';await this.api('/api/profiles/'+id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({...doc,id,name})});await this.init();this.q('profile-editor').value=id;await this.onEditorChange();this.flash('已创建独立预设','ok');}catch(e){this.flash('复制失败：'+e.message,'err');}
  }
  async makeDefault(){try{await this.api('/api/profiles/'+encodeURIComponent(this.q('profile-editor').value)+'/default',{method:'POST'});this.flash('已设置新任务的默认预设','ok');}catch(e){this.flash(e.message,'err');}}
  async deleteProfile(){const id=this.q('profile-editor').value;if(['current','construction','workstations'].includes(id))return this.flash('内置预设不能删除，可复制后修改','warn');try{await this.api('/api/profiles/'+encodeURIComponent(id),{method:'DELETE'});await this.init();this.flash('预设已删除，已有任务与历史记录保留其版本','ok');}catch(e){this.flash(e.message,'err');}}
}

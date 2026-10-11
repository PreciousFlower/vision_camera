/* Browser speakers; detection decisions are supplied only by the server's alerts. */
(function(root){
  'use strict';
  class AlertSpeech {
    constructor({synth=root.speechSynthesis, Utterance=root.SpeechSynthesisUtterance,
                 onStatus=()=>{}, storage=root.sessionStorage, now=()=>Date.now()}={}){
      this.synth=synth;this.Utterance=Utterance;this.onStatus=onStatus;
      this.storage=storage;this.now=now;this.enabled=false;this.volume=1;
      this.queue=[];this.active=null;this.timer=null;this.sequence=0;
      this.seen=new Set();
      try { const saved=JSON.parse(storage.getItem('civility-spoken-alerts')||'[]');
        if(Array.isArray(saved))saved.slice(-2000).forEach(key=>this.seen.add(key));
      } catch(e) { /* Storage restrictions must not disable detection. */ }
    }
    get supported(){return !!(this.synth&&this.Utterance);}
    key(job,a,live=false){
      const evidenceTime=a.evidence_time_sec;
      const time=live&&typeof evidenceTime==='number'&&Number.isFinite(evidenceTime)&&evidenceTime>=0?evidenceTime:Number(a.t_start)||0;
      return JSON.stringify([job,a.behavior_id||a.name,time,a.severity||'medium']);
    }
    alerts(values){return Array.isArray(values)?values.filter(a=>a&&typeof a==='object'&&typeof a.behavior_id==='string'&&a.behavior_id&&Number.isFinite(Number(a.t_start))):[];}
    remember(key){
      this.seen.add(key);while(this.seen.size>2000)this.seen.delete(this.seen.values().next().value);
      try{this.storage.setItem('civility-spoken-alerts',JSON.stringify([...this.seen]));}catch(e){}
    }
    seed(job,alerts,{live=false}={}){for(const a of this.alerts(alerts))this.remember(this.key(job,a,live));}
    begin(job){
      for(const key of this.seen){try{if(JSON.parse(key)[0]===job)this.seen.delete(key);}catch(e){this.seen.delete(key);}}
      this.queue=this.queue.filter(item=>item.job!==job);
    }
    enable(value){
      this.enabled=!!value&&this.supported;
      if(!this.enabled){this.queue=[];this.sequence++;clearTimeout(this.timer);this.active=null;if(this.synth)this.synth.cancel();}
      this.onStatus(this.supported?(this.enabled?'语音告警已启用':'语音告警已关闭'):'浏览器不支持语音，请使用 Edge 或 Chrome');
      return this.enabled;
    }
    test(text='语音播报测试，检测到告警后将在此音箱播放。'){
      if(!this.supported)return this.onStatus('当前浏览器不支持语音播报');
      const utterance=this.utterance(text);utterance.onerror=()=>this.onStatus('语音播放失败，请检查浏览器声音权限和系统音量');
      this.synth.speak(utterance);
    }
    utterance(text){
      const u=new this.Utterance(text);u.lang='zh-CN';u.volume=this.volume;u.rate=1;
      const voices=this.synth.getVoices();
      u.voice=voices.find(v=>/^zh[-_]CN/i.test(v.lang))||voices.find(v=>/^zh/i.test(v.lang))||null;
      return u;
    }
    notify(job,alerts,{demo=false,live=false}={}){
      for(const a of this.alerts(alerts)){
        const key=this.key(job,a,live);if(this.seen.has(key))continue;this.remember(key);
        if(!this.enabled)continue;
        const name=String(a.behavior_name||a.name||a.behavior_id||'现场风险').slice(0,40);
        const reason=String(a.reason||a.alert_reason||a.observable||'').slice(0,80);
        const time=Number.isFinite(Number(a.t_start))?`视频${Number(a.t_start).toFixed(1)}秒，`:'';
        this.queue.push({job,created:this.now(),text:`${demo?'演示告警，模拟结论。':'告警。'}${live?'':time}${name}。${reason}`});
      }
      this.queue=this.queue.slice(-8);this.drain();
    }
    drain(){
      if(this.active||!this.enabled)return;
      this.queue=this.queue.filter(item=>this.now()-item.created<20000);
      const item=this.queue.shift();if(!item)return;
      this.active=item;const sequence=++this.sequence,u=this.utterance(item.text);
      const finish=(failed=false)=>{
        if(sequence!==this.sequence)return;
        this.sequence++;clearTimeout(this.timer);this.active=null;
        if(failed){this.enable(false);this.onStatus('语音播放失败，请检查声音权限、音箱和系统音量，再重新启用');}
        else{this.onStatus('语音告警已启用');this.drain();}
      };
      u.onend=()=>finish();u.onerror=()=>finish(true);
      this.timer=setTimeout(()=>{if(sequence!==this.sequence)return;this.synth.cancel();finish(true);},30000);
      this.onStatus('正在播报：'+item.text);
      try{this.synth.speak(u);}catch(e){finish(true);}
    }
  }
  root.AlertSpeech=AlertSpeech;
  if(typeof module!=='undefined'&&module.exports)module.exports=AlertSpeech;
})(typeof window==='undefined'?globalThis:window);

/* Browser speakers; detection decisions are supplied only by the server's alerts. */
(function(root){
  'use strict';
  class AlertSpeech {
    static isMandarinVoice(voice){
      if(!voice||typeof voice.lang!=='string')return false;
      const lang=voice.lang.replace(/_/g,'-').toLowerCase();
      const name=String(voice.name||'');
      if(/(?:^|-)yue(?:-|$)|(?:^|-)(hk|tw|mo)(?:-|$)|hant/.test(lang)
          ||/cantonese|hong\s*kong|taiwan|粤语|廣東話|广东话|香港|台灣|台湾/i.test(name))return false;
      return /^(?:zh|cmn)-(?:cn|hans(?:-cn)?)$|^zh-cmn-hans-cn$/.test(lang);
    }
    constructor({synth=root.speechSynthesis, Utterance=root.SpeechSynthesisUtterance,
                 onStatus=()=>{}, storage=root.sessionStorage, now=()=>Date.now(),
                 fetchAudio=root.fetch?.bind(root), makeAudio=root.Audio?()=>new root.Audio():null,
                 urls=root.URL}={}){
      this.synth=synth;this.Utterance=Utterance;this.onStatus=onStatus;
      this.storage=storage;this.now=now;this.enabled=false;this.volume=1;
      this.queue=[];this.active=null;this.timer=null;this.sequence=0;
      this.seen=new Set();
      this.voice='auto';this.rate=1;this.fetchAudio=fetchAudio;this.makeAudio=makeAudio;this.urls=urls;
      this.audio=null;this.audioUrl=null;this.controller=null;this.requestTimer=null;this.neuralRetryAt=0;this.lastEngine='标准普通话 · 系统语音';
      this.fallbackSequence=0;
      try { const saved=JSON.parse(storage.getItem('civility-spoken-alerts')||'[]');
        if(Array.isArray(saved))saved.slice(-2000).forEach(key=>this.seen.add(key));
      } catch(e) { /* Storage restrictions must not disable detection. */ }
    }
    get supported(){return !!((this.synth&&this.Utterance)||(this.fetchAudio&&this.makeAudio));}
    configure({voice=this.voice,rate=this.rate,volume=this.volume}={}){
      this.voice=typeof voice==='string'&&(voice==='auto'||/^neural:(xiaoxiao|yunxi)$/.test(voice)||voice.startsWith('browser:'))?voice:'auto';
      this.rate=Number.isFinite(Number(rate))?Math.max(.85,Math.min(1.15,Number(rate))):1;
      this.volume=Number.isFinite(Number(volume))?Math.max(0,Math.min(1,Number(volume))):1;
    }
    clearAudio(){
      clearTimeout(this.requestTimer);this.requestTimer=null;
      if(this.controller){this.controller.abort();this.controller=null;}
      if(this.audio){this.audio.onended=null;this.audio.onerror=null;this.audio.pause();this.audio.removeAttribute?.('src');this.audio=null;}
      if(this.audioUrl){this.urls.revokeObjectURL(this.audioUrl);this.audioUrl=null;}
    }
    key(job,a,live=false){
      const evidenceTime=a.evidence_time_sec;
      const time=live&&typeof evidenceTime==='number'&&Number.isFinite(evidenceTime)&&evidenceTime>=0?evidenceTime:Number(a.t_start)||0;
      return JSON.stringify([job,a.behavior_id||a.name,time,a.severity||'medium']);
    }
    alerts(values){return Array.isArray(values)?values.filter(a=>a&&typeof a==='object'&&typeof a.behavior_id==='string'&&a.behavior_id&&Number.isFinite(Number(a.t_start))):[];}
    speakable(a){return a.review_status!=='rejected'&&!['withdrawn','resolved'].includes(a.lifecycle);}
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
      if(!this.enabled){this.queue=[];this.sequence++;clearTimeout(this.timer);this.active=null;this.clearAudio();if(this.synth)this.synth.cancel();}
      this.onStatus(this.supported?(this.enabled?'语音告警已启用':'语音告警已关闭'):'浏览器不支持语音，请使用 Edge 或 Chrome');
      return this.enabled;
    }
    test(text='你好，我会提醒你关注现场，请保持留意。'){
      if(!this.supported)return this.onStatus('当前浏览器不支持语音播报');
      const interrupted=this.active&&!this.active.preview?this.active:null;
      // Every click owns a fresh playback generation. Late speech/network
      // callbacks cannot complete or play over the newest preview.
      this.sequence++;clearTimeout(this.timer);this.timer=null;
      this.active=null;this.clearAudio();this.synth?.cancel();
      this.queue=this.queue.filter(item=>!item.preview);
      if(interrupted)this.queue.unshift(interrupted);
      this.queue.unshift({text,created:this.now(),preview:true});this.drain();
    }
    utterance(text){
      const u=new this.Utterance(text);u.lang='zh-CN';u.volume=this.volume;u.rate=this.rate;u.pitch=1;
      const chinese=(this.synth.getVoices()||[]).filter(AlertSpeech.isMandarinVoice);
      const preferred=this.voice.startsWith('browser:')?this.voice.slice(8):'';
      u.voice=preferred?chinese.find(v=>(v.voiceURI||v.name)===preferred)
        :chinese.find(v=>v.localService===true)||chinese[0];
      if(!u.voice)return null;
      return u;
    }
    playBrowser(text,sequence,finish){
      if(sequence!==this.sequence)return;
      if(!this.synth||!this.Utterance)return finish(true);
      const u=this.utterance(text);
      if(!u)return finish(true,this.voice.startsWith('browser:')
        ?'所选声音不是可用的标准普通话，请重新选择标准普通话声音'
        :'未找到标准普通话声音，请在系统语音设置中安装中文（简体，中国）声音后重试');
      u.onend=()=>finish();u.onerror=()=>finish(true);
      this.lastEngine='标准普通话 · 系统语音';
      try{this.synth.speak(u);}catch(e){finish(true);}
    }
    fallback(text,sequence,finish){
      if(sequence!==this.sequence||this.fallbackSequence===sequence)return;
      this.fallbackSequence=sequence;this.clearAudio();this.neuralRetryAt=this.now()+60000;
      this.onStatus('自然语音暂不可用，本次使用标准普通话系统声音');this.playBrowser(text,sequence,finish);
    }
    async playNeural(text,sequence,finish){
      if(!this.fetchAudio||!this.makeAudio)throw new Error('Audio unavailable');
      const controller=new AbortController();this.controller=controller;
      const timer=setTimeout(()=>controller.abort(),10500);this.requestTimer=timer;
      let blob;
      try{
        const response=await this.fetchAudio('/api/speech/synthesize',{method:'POST',signal:controller.signal,
          headers:{'Content-Type':'application/json'},body:JSON.stringify({text,voice:this.voice.slice(7),rate:this.rate})});
        if(!response.ok||!/^audio\/(mpeg|wav)/i.test(response.headers.get('Content-Type')||''))throw new Error('Neural speech unavailable');
        blob=await response.blob();if(!blob.size)throw new Error('Empty audio');
      }finally{clearTimeout(timer);if(this.requestTimer===timer)this.requestTimer=null;}
      if(sequence!==this.sequence)return;
      this.audioUrl=this.urls.createObjectURL(blob);const audio=this.makeAudio();this.audio=audio;
      audio.volume=this.volume;audio.src=this.audioUrl;this.lastEngine='自然语音';
      audio.onended=()=>finish();audio.onerror=()=>this.fallback(text,sequence,finish);
      await audio.play();
    }
    notify(job,alerts,{demo=false,live=false}={}){
      const values=this.alerts(alerts);
      const current=new Set(values.filter(a=>this.speakable(a)).map(a=>this.key(job,a,live)));
      const ended=new Set(values.filter(a=>!this.speakable(a)).map(a=>this.key(job,a,live)));
      const cancelled=item=>ended.has(item.key)||(item.provisional&&!current.has(item.key));
      this.queue=this.queue.filter(item=>!(item.job===job&&cancelled(item)));
      if(this.active?.job===job&&cancelled(this.active)){
        this.sequence++;clearTimeout(this.timer);this.active=null;this.clearAudio();this.synth?.cancel();
        this.onStatus('告警已结束或撤回，已取消这条语音');
      }
      for(const a of values){
        const key=this.key(job,a,live);if(this.seen.has(key))continue;this.remember(key);
        if(!this.enabled||!this.speakable(a))continue;
        const name=String(a.behavior_name||a.name||a.behavior_id||'现场风险').slice(0,40);
        const reason=String(a.reason||a.alert_reason||a.observable||'').slice(0,80);
        const at=a.source==='workstation'&&Number.isFinite(Number(a.threshold_time_sec))?a.threshold_time_sec:a.t_start;
        const time=Number.isFinite(Number(at))?`视频${Number(at).toFixed(1)}秒，`:'';
        const provisional=a.source==='workstation'&&a.review_status==='pending';
        this.queue.push({job,key,provisional,created:this.now(),text:`${demo?'演示告警，模拟结论。':'请注意，'}${live?'':time}${provisional?'已到设定阈值，'+name+'待AI复核，请核查现场。':'检测到'+name+'。'}${reason}`});
      }
      this.queue=this.queue.slice(-8);this.drain();
    }
    drain(){
      if(this.active||(!this.enabled&&!this.queue[0]?.preview))return;
      this.queue=this.queue.filter(item=>this.now()-item.created<20000);
      const item=this.queue.shift();if(!item)return;
      this.active=item;const sequence=++this.sequence;
      const finish=(failed=false,message='')=>{
        if(sequence!==this.sequence)return;
        this.sequence++;clearTimeout(this.timer);this.clearAudio();this.active=null;
        if(failed){this.enable(false);this.onStatus(message||'语音播放失败，请检查声音权限、音箱和系统音量，再重新启用');}
        else{this.onStatus((item.preview?'试听完成':'语音告警已启用')+' · '+this.lastEngine);this.drain();}
      };
      this.timer=setTimeout(()=>{if(sequence!==this.sequence)return;this.synth?.cancel();finish(true);},30000);
      this.onStatus('正在播报：'+item.text);
      if(!item.preview&&this.voice.startsWith('neural:')&&this.now()>=this.neuralRetryAt){
        this.playNeural(item.text,sequence,finish).catch(()=>{
          this.fallback(item.text,sequence,finish);
        });
      }else this.playBrowser(item.text,sequence,finish);
    }
  }
  root.AlertSpeech=AlertSpeech;
  if(typeof module!=='undefined'&&module.exports)module.exports=AlertSpeech;
})(typeof window==='undefined'?globalThis:window);

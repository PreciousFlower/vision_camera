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
    static voiceGender(voice){
      if(!AlertSpeech.isMandarinVoice(voice))return null;
      // Web Speech does not expose a gender field. Only voices with verified
      // Mainland Mandarin identities may satisfy an explicit gender choice.
      const name=String(voice.name||'')+' '+String(voice.voiceURI||'');
      if(/\b(?:huihui|yaoyao|xiaoxiao|xiaoyi)\b|(?:惠惠|慧慧|瑶瑶|曉曉|晓晓|晓伊)/i.test(name))return 'female';
      if(/\b(?:kangkang|yunxi|yunyang|yunjian)\b|(?:康康|云希|雲希|云扬|雲揚|云健)/i.test(name))return 'male';
      return null;
    }
    constructor({synth=root.speechSynthesis, Utterance=root.SpeechSynthesisUtterance,
                 onStatus=()=>{}, onEnabledChange=()=>{}, storage=root.sessionStorage, now=()=>Date.now(),
                 fetchAudio=root.fetch?.bind(root), makeAudio=root.Audio?()=>new root.Audio():null,
                 urls=root.URL}={}){
      this.synth=synth;this.Utterance=Utterance;this.onStatus=onStatus;this.onEnabledChange=onEnabledChange;
      this.storage=storage;this.now=now;this.enabled=false;this.volume=1;
      this.queue=[];this.active=null;this.timer=null;this.sequence=0;
      this.seen=new Set();
      this.voice='auto';this.rate=1;this.fetchAudio=fetchAudio;this.makeAudio=makeAudio;this.urls=urls;
      this.audio=null;this.audioUrl=null;this.controller=null;this.requestTimer=null;this.neuralRetryAt=0;this.lastEngine='标准普通话 · 系统语音';
      this.fallbackSequence=0;this.retryTimer=null;this.waitingForVoices=false;this.voiceWaitStarted=null;this.lastError='';
      this.resumeVoices=()=>{if(this.waitingForVoices){this.resumeVoiceQueue();this.drain();}};
      this.synth?.addEventListener?.('voiceschanged',this.resumeVoices);
      try { const saved=JSON.parse(storage.getItem('civility-spoken-alerts')||'[]');
        if(Array.isArray(saved))saved.slice(-2000).forEach(key=>this.seen.add(key));
      } catch(e) { /* Storage restrictions must not disable detection. */ }
    }
    get supported(){return !!((this.synth&&this.Utterance)||(this.fetchAudio&&this.makeAudio));}
    configure({voice=this.voice,rate=this.rate,volume=this.volume}={}, {restart=true}={}){
      const previousVoice=this.voice,previousRate=this.rate;
      this.voice=typeof voice==='string'&&(voice==='auto'||/^mandarin:(female|male)$/.test(voice)||/^neural:(xiaoxiao|yunxi)$/.test(voice)||voice.startsWith('browser:'))?voice:'auto';
      this.rate=Number.isFinite(Number(rate))?Math.max(.6,Math.min(1.6,Number(rate))):1;
      this.volume=Number.isFinite(Number(volume))?Math.max(0,Math.min(1,Number(volume))):1;
      if(this.audio)this.audio.volume=this.volume;
      const changed=previousVoice!==this.voice||previousRate!==this.rate;
      const current=restart&&changed?this.active:null;
      if(current){
        // Invalidate callbacks before cancel(): some engines synchronously
        // emit "interrupted" while a replacement utterance is being created.
        this.sequence++;clearTimeout(this.timer);this.timer=null;
        clearTimeout(this.retryTimer);this.retryTimer=null;
        this.active=null;this.clearAudio();this.synth?.cancel();
        this.resumeVoiceQueue();this.queue.unshift({...current,created:this.now()});this.drain();
      }
      return {changed,restarted:!!current};
    }
    resumeVoiceQueue(){
      if(this.voiceWaitStarted!==null){
        const paused=Math.max(0,this.now()-this.voiceWaitStarted);
        this.queue.forEach(item=>item.created+=paused);
      }
      this.waitingForVoices=false;this.voiceWaitStarted=null;
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
      const previous=this.enabled;
      this.enabled=!!value&&this.supported;
      if(!this.enabled){this.queue=[];this.sequence++;clearTimeout(this.timer);clearTimeout(this.retryTimer);this.retryTimer=null;this.waitingForVoices=false;this.voiceWaitStarted=null;this.active=null;this.clearAudio();if(this.synth)this.synth.cancel();}
      if(previous!==this.enabled)this.onEnabledChange(this.enabled);
      this.onStatus(this.supported?(this.enabled?'语音告警已启用':'语音告警已关闭'):'浏览器不支持语音，请使用 Edge 或 Chrome');
      return this.enabled;
    }
    test(text='你好，我会提醒你关注现场，请保持留意。'){
      if(!this.supported)return this.onStatus('当前浏览器不支持语音播报');
      const interrupted=this.active&&!this.active.preview?this.active:null;
      // Every click owns a fresh playback generation. Late speech/network
      // callbacks cannot complete or play over the newest preview.
      this.sequence++;clearTimeout(this.timer);this.timer=null;
      clearTimeout(this.retryTimer);this.retryTimer=null;this.resumeVoiceQueue();
      this.active=null;this.clearAudio();this.synth?.cancel();
      this.queue=this.queue.filter(item=>!item.preview);
      if(interrupted)this.queue.unshift({...interrupted,created:this.now()});
      this.queue.unshift({text,created:this.now(),preview:true});this.drain();
    }
    utterance(text){
      const u=new this.Utterance(text);u.lang='zh-CN';u.volume=this.volume;u.rate=this.rate;u.pitch=1;
      const chinese=(this.synth.getVoices()||[]).filter(AlertSpeech.isMandarinVoice);
      const preferred=this.voice.startsWith('browser:')?this.voice.slice(8):'';
      const gender=this.voice.startsWith('mandarin:')?this.voice.slice(9):'';
      const matching=gender?chinese.filter(v=>AlertSpeech.voiceGender(v)===gender):chinese;
      u.voice=preferred?chinese.find(v=>(v.voiceURI||v.name)===preferred)
        :matching.find(v=>v.localService===true)||matching[0];
      if(!u.voice)return null;
      return u;
    }
    playBrowser(text,sequence,finish,attempt=0){
      if(sequence!==this.sequence)return;
      if(!this.synth||!this.Utterance)return finish(true);
      const u=this.utterance(text);
      if(!u){
        const gender=this.voice==='mandarin:male'?'男声':this.voice==='mandarin:female'?'女声':'';
        return finish(true,gender?'暂未找到标准普通话'+gender+'，请等待声音加载或在系统语音设置安装对应声音后重试'
          :this.voice.startsWith('browser:')?'所选标准普通话声音暂不可用，请等待声音加载或重新选择'
          :'暂未找到标准普通话声音，请等待声音加载或在系统语音设置安装中文（简体，中国）声音后重试','voices-loading');
      }
      if(this.waitingForVoices)this.resumeVoiceQueue();
      let settled=false;
      const failed=event=>{
        if(sequence!==this.sequence||settled)return;
        settled=true;
        const code=String(event?.error||'synthesis-failed');
        this.lastError=code;
        if(['audio-busy','synthesis-failed','interrupted','canceled'].includes(code)&&attempt<2){
          // Cancellation may briefly leave a native voice busy. Retrying the
          // same verified voice preserves the requested gender and dialect.
          this.retryTimer=setTimeout(()=>{this.retryTimer=null;this.playBrowser(text,sequence,finish,attempt+1);},80*(attempt+1));
          return;
        }
        const messages={
          'canceled':'播报已被浏览器取消，告警已保留，可点击试听继续',
          'interrupted':'播报暂时被打断，告警已保留，可点击试听继续',
          'audio-busy':'音频通道暂时忙碌，告警已保留，请稍后点击试听重试',
          'voice-unavailable':'所选普通话声音暂不可用，告警已保留，请等待声音加载或重新选择',
          'language-unavailable':'标准普通话声音尚未就绪，告警已保留，请安装中文（简体，中国）声音',
          'not-allowed':'浏览器未允许播放语音，请点击试听检查后重新勾选“告警语音播报”',
          'audio-hardware':'音频设备不可用，请检查音箱和系统输出，试听正常后重新勾选“告警语音播报”',
          'invalid-argument':'声音或语速设置暂不受支持，请重新选择后试听',
          'synthesis-unavailable':'系统语音服务暂不可用，告警已保留，请稍后重试',
          'synthesis-failed':'系统语音暂未播放成功，告警已保留，请重新选择声音或点击试听'
        };
        finish(true,messages[code]||'语音暂未播放成功，告警已保留，请点击试听重试',code);
      };
      u.onend=()=>{if(!settled){settled=true;finish();}};u.onerror=failed;
      const gender=AlertSpeech.voiceGender(u.voice);
      this.lastEngine='标准普通话'+(gender==='male'?'男声':gender==='female'?'女声':'')+' · 系统语音';
      try{this.synth.speak(u);}catch(e){failed({error:e?.name==='NotAllowedError'?'not-allowed':'synthesis-failed'});}
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
          headers:{'Content-Type':'application/json'},body:JSON.stringify({text,voice:this.voice.slice(7),rate:Math.max(.85,Math.min(1.15,this.rate))})});
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
        this.sequence++;clearTimeout(this.timer);clearTimeout(this.retryTimer);this.retryTimer=null;this.active=null;this.clearAudio();this.synth?.cancel();
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
      if(!this.waitingForVoices)this.queue=this.queue.filter(item=>this.now()-item.created<20000);
      const item=this.queue.shift();if(!item)return;
      this.active=item;const sequence=++this.sequence;
      const finish=(failed=false,message='',code='')=>{
        if(sequence!==this.sequence)return;
        this.sequence++;clearTimeout(this.timer);clearTimeout(this.retryTimer);this.retryTimer=null;this.clearAudio();this.active=null;
        if(failed){
          this.lastError=code||'synthesis-failed';
          // Voice discovery and transient native errors do not switch off an
          // enabled alarm or discard genuine pending announcements.
          if(!item.preview||code==='voices-loading')this.queue.unshift({...item,created:this.now()});
          this.waitingForVoices=['voices-loading','voice-unavailable','language-unavailable'].includes(code);
          if(this.waitingForVoices&&this.voiceWaitStarted===null)this.voiceWaitStarted=this.now();
          if(['not-allowed','audio-hardware'].includes(code)&&this.enabled){this.enabled=false;this.onEnabledChange(false);}
          this.onStatus(message||'语音暂未播放成功，告警已保留，请点击试听重试');
        }
        else{this.lastError='';this.onStatus((item.preview?'试听完成':'语音告警已启用')+' · '+this.lastEngine);this.drain();}
      };
      const timeout=Math.max(30000,Math.min(90000,Math.ceil(String(item.text).length*300/this.rate)));
      this.timer=setTimeout(()=>{if(sequence!==this.sequence)return;finish(true,'播报等待超时，告警已保留，请点击试听重试','timeout');this.synth?.cancel();},timeout);
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

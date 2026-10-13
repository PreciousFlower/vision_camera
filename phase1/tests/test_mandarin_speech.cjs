const assert=require('node:assert/strict');
const AlertSpeech=require('../webui/static/alert_speech.js');
const flush=()=>new Promise(resolve=>setImmediate(resolve));
const storage={getItem:()=>null,setItem:()=>{}};
class Utterance{constructor(text){this.text=text;}}
function harness(voices){
  const spoken=[],statuses=[];let cancelled=0;
  const synth={getVoices:()=>voices,speak:u=>spoken.push(u),cancel:()=>cancelled++};
  const speaker=new AlertSpeech({synth,Utterance,storage,onStatus:s=>statuses.push(s)});
  return {speaker,spoken,statuses,get cancelled(){return cancelled;}};
}
const voices=[
  {name:'Hong Kong Cantonese Natural',lang:'zh-HK',voiceURI:'hk'},
  {name:'Taiwan Mandarin Natural',lang:'zh-TW',voiceURI:'tw'},
  {name:'Cantonese',lang:'yue-CN',voiceURI:'yue'},
  {name:'Microsoft Huihui',lang:'zh-CN',voiceURI:'cn',localService:true},
  {name:'Microsoft Xiaoxiao Natural',lang:'zh-CN',voiceURI:'cn-natural',localService:false}
];
async function main(){
  for(const lang of ['zh-CN','zh_CN','zh-Hans','zh-Hans-CN','cmn-CN','cmn-Hans-CN']){
    assert.equal(AlertSpeech.isMandarinVoice({lang,name:'Mandarin'}),true);
  }
  for(const lang of ['zh-HK','zh-TW','zh-Hant','yue-CN','yue-Hans-CN','en-US','zh']){
    assert.equal(AlertSpeech.isMandarinVoice({lang,name:'Natural Chinese'}),false);
  }
  assert.equal(AlertSpeech.isMandarinVoice({lang:'zh-CN',name:'Cantonese Natural'}),false);
  const auto=harness(voices);
  auto.speaker.test('标准普通话');
  assert.equal(auto.spoken[0].voice.voiceURI,'cn','Default must be a local standard Mainland Mandarin voice');
  assert.equal(auto.spoken[0].lang,'zh-CN');auto.speaker.enable(false);
  for(const uri of ['hk','tw','yue']){
    const selected=harness(voices);selected.speaker.configure({voice:'browser:'+uri});selected.speaker.test();
    assert.equal(selected.spoken.length,0,'Explicit non-Mandarin voice must not play');
    assert.match(selected.statuses.at(-1),/普通话/);selected.speaker.enable(false);
  }
  const missing=harness(voices.slice(0,3));missing.speaker.test();
  assert.equal(missing.spoken.length,0,'Missing Mandarin must not fall back to the browser default dialect');
  assert.match(missing.statuses.at(-1),/普通话/);
  const preview=harness(voices);preview.speaker.test('第一次');
  const first=preview.spoken[0];preview.speaker.test('第二次');
  assert.equal(preview.spoken.length,2,'Every preview click must immediately restart');
  assert.ok(preview.cancelled>=2);first.onend();
  assert.equal(preview.speaker.active.text,'第二次','Cancelled speech callbacks cannot finish the new preview');
  preview.spoken[1].onend();preview.speaker.enable(false);

  const alert={behavior_id:'station_away',behavior_name:'离岗超时',t_start:2};
  const interrupted=harness(voices);interrupted.speaker.enable(true);
  interrupted.speaker.notify('job',[alert,{...alert,t_start:3}]);
  const firstAlert=interrupted.spoken[0];interrupted.speaker.test('试听一次');
  assert.equal(interrupted.spoken.length,2,'Preview interrupts a playing alert');
  interrupted.speaker.test('试听两次');assert.equal(interrupted.spoken.length,3);
  assert.equal(interrupted.speaker.queue.filter(i=>!i.preview).length,2,'Interrupted and queued real alerts are preserved once');
  firstAlert.onend();interrupted.spoken[1].onend();
  assert.equal(interrupted.speaker.active.text,'试听两次');
  interrupted.spoken[2].onend();assert.equal(interrupted.spoken.length,4);
  assert.match(interrupted.spoken[3].text,/离岗超时/);
  interrupted.spoken[3].onend();assert.equal(interrupted.spoken.length,5);
  interrupted.spoken[4].onend();interrupted.speaker.enable(false);

  let release,signal;const played=[];
  const inflight=harness(voices);
  inflight.speaker.configure({voice:'neural:xiaoxiao'});
  inflight.speaker.fetchAudio=(_url,opts)=>{signal=opts.signal;return new Promise(resolve=>release=resolve);};
  inflight.speaker.makeAudio=()=>({play(){played.push(this);return Promise.resolve();},pause(){},removeAttribute(){}});
  inflight.speaker.urls={createObjectURL:()=> 'blob:test',revokeObjectURL:()=>{}};
  inflight.speaker.enable(true);inflight.speaker.notify('request',[alert]);
  inflight.speaker.test('立即试听');assert.equal(signal.aborted,true);
  assert.equal(inflight.speaker.requestTimer,null,'Preview cancels the old HTTP timeout as well');
  assert.equal(inflight.spoken.length,1,'Preview starts synchronously without waiting for a network synthesis');
  release({ok:true,headers:{get:()=> 'audio/wav'},blob:async()=>({size:100})});await flush();
  assert.equal(played.length,0,'A cancelled synthesis response cannot play after a new preview');
  assert.equal(inflight.speaker.active.text,'立即试听');inflight.speaker.enable(false);
  console.log('PASS Mainland Mandarin only; immediate repeated previews; real alert queue preserved; stale synthesis discarded');
}
main().catch(error=>{console.error(error);process.exit(1);});

const assert=require('node:assert/strict');
const AlertSpeech=require('../webui/static/alert_speech.js');
const storage={getItem:()=>null,setItem:()=>{}};
class Utterance{constructor(text){this.text=text;}}
const mainland=[
  {name:'Microsoft Huihui - Chinese (Simplified, PRC)',lang:'zh-CN',voiceURI:'female',localService:true},
  {name:'Microsoft Kangkang - Chinese (Simplified, PRC)',lang:'zh-CN',voiceURI:'male',localService:true},
  {name:'Microsoft Xiaoxiao Online (Natural)',lang:'zh-CN',voiceURI:'online-female',localService:false},
  {name:'Hong Kong Cantonese',lang:'zh-HK',voiceURI:'hk',localService:true}
];
function harness(voices=mainland,now=()=>Date.now()){
  const spoken=[],statuses=[];let cancelled=0,current=null;
  const listeners=new Map();
  const synth={getVoices:()=>voices,speak:u=>{current=u;spoken.push(u);},cancel(){
    cancelled++;const previous=current;current=null;previous?.onerror?.({error:'interrupted'});
  },addEventListener(type,callback){listeners.set(type,callback);}};
  const speaker=new AlertSpeech({synth,Utterance,storage,now,onStatus:s=>statuses.push(s)});
  return {speaker,spoken,statuses,synth,setVoices(value){voices=value;},voicesChanged(){listeners.get('voiceschanged')?.();},get cancelled(){return cancelled;}};
}
const alert={behavior_id:'smoking',behavior_name:'吸烟',t_start:1};
let failures=0;
function check(name,run){let h;try{run(value=>h=value);console.log('PASS '+name);}catch(error){
  failures++;console.error('FAIL '+name+': '+error.message);
}finally{h?.speaker.enable(false);}}
check('active female preview restarts immediately in a male Mandarin voice',keep=>{
  const h=harness();keep(h);h.speaker.configure({voice:'mandarin:female'});h.speaker.test('现场提醒');
  const stale=h.spoken[0];h.speaker.configure({voice:'mandarin:male'});
  assert.equal(h.spoken.length,2,'Changing voice must restart the current preview');
  assert.equal(h.spoken[1].voice.voiceURI,'male');stale.onerror({error:'canceled'});
  assert.equal(h.speaker.active.text,'现场提醒','Old cancellation cannot finish the new voice');
});
check('rate change immediately restarts actual spoken audio at requested 0.75 and 1.5',keep=>{
  const h=harness();keep(h);h.speaker.test('调整语速');h.speaker.configure({rate:.75});
  assert.equal(h.spoken.length,2,'Changing rate must restart the current speech');assert.equal(h.spoken[1].rate,.75);
  h.speaker.configure({rate:1.5});assert.equal(h.spoken[2].rate,1.5);
});
check('temporary empty voice inventory does not turn off enabled alerts or discard queued alarms',keep=>{
  const h=harness([]);keep(h);h.speaker.enable(true);h.speaker.notify('empty',[alert,{...alert,t_start:2}]);
  assert.equal(h.speaker.enabled,true,'Loading voices is recoverable, not a hardware failure');
  assert.equal(h.speaker.queue.length,2,'Both genuine alarms must remain available when voices load');
  h.setVoices(mainland);h.speaker.drain();assert.equal(h.spoken.length,1);
});
check('canceled current utterance is handled without a false speaker-permission failure',keep=>{
  const h=harness();keep(h);h.speaker.enable(true);h.speaker.notify('cancel',[alert,{...alert,t_start:2}]);
  h.spoken[0].onerror({error:'canceled'});
  assert.equal(h.speaker.enabled,true,'Browser cancellation must not disable voice alerts');
  assert.ok(h.statuses.every(s=>!s.includes('检查声音权限')));
});
check('unknown Mandarin gender never pretends to be a male voice',keep=>{
  const h=harness([{name:'Unknown Mandarin',lang:'zh-CN',voiceURI:'unknown'}]);keep(h);
  h.speaker.configure({voice:'mandarin:male'});h.speaker.test('男性普通话');
  assert.equal(h.spoken.length,0,'Unknown gender cannot satisfy a male selection');assert.match(h.statuses.at(-1),/男/);
});
check('voice loading resumes real announcements even after the normal queue expiry',keep=>{
  let now=0;const h=harness([],()=>now);keep(h);h.speaker.enable(true);
  h.speaker.notify('loading',[alert,{...alert,t_start:2}]);now=25000;h.setVoices(mainland);h.voicesChanged();
  assert.equal(h.spoken.length,1);h.spoken[0].onend();assert.equal(h.spoken.length,2,'Waiting for a voice cannot silently expire the second real alarm');
});
check('active alert switching retains queued alarms and volume-only changes do not restart',keep=>{
  const h=harness();keep(h);h.speaker.enable(true);h.speaker.notify('live',[alert,{...alert,t_start:2}]);
  h.speaker.configure({voice:'mandarin:male',rate:1.3});assert.equal(h.spoken.length,2);assert.equal(h.spoken[1].voice.voiceURI,'male');
  assert.equal(h.speaker.queue.length,1);const generation=h.speaker.sequence;
  h.speaker.configure({volume:.4});assert.equal(h.spoken.length,2);assert.equal(h.speaker.sequence,generation);
  h.spoken[1].onend();assert.equal(h.spoken.length,3);assert.match(h.spoken[2].text,/2.0秒/);
});
check('unavailable explicit voice preserves selection and never plays a different gender',keep=>{
  const h=harness(mainland.filter(v=>AlertSpeech.voiceGender(v)==='female'));keep(h);h.speaker.enable(true);
  h.speaker.configure({voice:'mandarin:male'});h.speaker.notify('absent',[alert]);
  assert.equal(h.spoken.length,0);assert.equal(h.speaker.voice,'mandarin:male');assert.equal(h.speaker.enabled,true);
  h.setVoices(mainland);h.voicesChanged();assert.equal(h.spoken[0].voice.voiceURI,'male');
});
check('actual hardware failure reports disabled state to the UI while keeping the genuine alert',keep=>{
  const h=harness();keep(h);const enabled=[];h.speaker.onEnabledChange=value=>enabled.push(value);
  h.speaker.enable(true);h.speaker.notify('hardware',[alert]);h.spoken[0].onerror({error:'audio-hardware'});
  assert.deepEqual(enabled,[true,false]);assert.equal(h.speaker.enabled,false);assert.equal(h.speaker.queue.length,1);
  assert.equal(h.speaker.lastError,'audio-hardware');assert.match(h.statuses.at(-1),/设备/);
});
check('switching a long active alert restarts it while older unplayed backlog still expires',keep=>{
  let now=0;const h=harness(mainland,()=>now);keep(h);h.speaker.enable(true);
  h.speaker.notify('long-active',[alert,{...alert,t_start:2}]);now=25000;
  h.speaker.configure({voice:'mandarin:male',rate:.75});
  assert.equal(h.spoken.length,2,'An explicitly restarted current alert must not be dropped as stale');
  assert.equal(h.spoken[1].voice.voiceURI,'male');assert.equal(h.spoken[1].rate,.75);
  assert.equal(h.speaker.active.job,'long-active');assert.equal(h.speaker.queue.length,0,'Only the explicitly restarted active alert gets a new freshness time');
});
check('preview interruption resumes a long current alert without resurrecting stale backlog',keep=>{
  let now=0;const h=harness(mainland,()=>now);keep(h);h.speaker.enable(true);
  h.speaker.notify('long-preview',[alert,{...alert,t_start:2}]);now=25000;
  h.speaker.test('试听音箱');assert.equal(h.spoken.length,2);h.spoken[1].onend();
  assert.equal(h.spoken.length,3,'The long current alert must resume after preview');
  assert.equal(h.speaker.active.job,'long-preview');assert.equal(h.speaker.queue.length,0);
});
check('recovering a failed long current alert preserves it without replaying expired backlog',keep=>{
  let now=0;const h=harness(mainland,()=>now);keep(h);h.speaker.enable(true);
  h.speaker.notify('long-failure',[alert,{...alert,t_start:2}]);now=25000;
  h.spoken[0].onerror({error:'audio-hardware'});now=35000;h.speaker.enable(true);h.speaker.test('恢复试听');
  h.spoken[1].onend();assert.equal(h.spoken.length,3,'The failed active alert remains recoverable after its original freshness period');
  assert.equal(h.speaker.active.job,'long-failure');assert.equal(h.speaker.queue.length,0);
});
async function transientRetry(){
  const h=harness();try{
    h.speaker.configure({voice:'mandarin:male'});h.speaker.enable(true);h.speaker.notify('retry',[alert]);
    h.spoken[0].onerror({error:'audio-busy'});
    h.spoken[0].onerror({error:'audio-busy'});h.spoken[0].onend();
    await new Promise(resolve=>setTimeout(resolve,100));
    assert.equal(h.spoken.length,2);assert.equal(h.spoken[1].voice.voiceURI,'male');
    h.spoken[1].onerror({error:'synthesis-failed'});await new Promise(resolve=>setTimeout(resolve,180));
    assert.equal(h.spoken.length,3);assert.equal(h.spoken[2].voice.voiceURI,'male');
    h.spoken[2].onerror({error:'interrupted'});await new Promise(resolve=>setTimeout(resolve,180));
    assert.equal(h.spoken.length,3,'Transient errors may retry twice, never spin indefinitely');
    assert.equal(h.speaker.enabled,true);assert.equal(h.speaker.queue.length,1);
    assert.ok(h.statuses.every(s=>!s.includes('检查声音权限')));
    console.log('PASS bounded transient retry preserves Mandarin gender and queued alert');
  }catch(error){failures++;console.error('FAIL bounded transient retry: '+error.message);}
  finally{h.speaker.enable(false);if(failures)process.exitCode=1;}
}
transientRetry();

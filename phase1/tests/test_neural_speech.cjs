const assert=require('node:assert/strict');
const AlertSpeech=require('../webui/static/alert_speech.js');
const flush=()=>new Promise(resolve=>setImmediate(resolve));
const alert={behavior_id:'littering',behavior_name:'乱扔垃圾',t_start:0,reason:'可见物品落地'};
async function main(){
  const spoken=[],played=[],revoked=[],storage={getItem:()=>null,setItem:()=>{}};
  class U{constructor(text){this.text=text;}}
  const synth={getVoices:()=>[{name:'传统',lang:'zh-CN'},{name:'Natural Chinese',lang:'zh-CN',voiceURI:'natural'}],
    speak:u=>spoken.push(u),cancel:()=>{}};
  const urls={createObjectURL:()=> 'blob:test',revokeObjectURL:url=>revoked.push(url)};
  class Audio{play(){played.push(this);return Promise.resolve();}pause(){}removeAttribute(){}}
  const response={ok:true,headers:{get:()=> 'audio/wav'},blob:async()=>({size:123})};
  const speaker=new AlertSpeech({synth,Utterance:U,storage,fetchAudio:async()=>response,makeAudio:()=>new Audio(),urls});
  speaker.configure({voice:'neural:xiaoxiao',rate:.9,volume:.4});speaker.enable(true);speaker.notify('one',[alert]);
  await flush();assert.equal(played.length,1);assert.equal(spoken.length,0);assert.equal(played[0].volume,.4);
  speaker.test();assert.equal(played.length,1); // Preview waits for the alert.
  played[0].onended();await flush();assert.equal(played.length,2);played[1].onended();assert.equal(revoked.length,2);
  speaker.fetchAudio=async()=>{throw new Error('offline');};speaker.notify('two',[alert]);await flush();
  assert.equal(spoken.length,1);assert.equal(spoken[0].voice.voiceURI,'natural');assert.equal(spoken[0].rate,.9);spoken[0].onend();
  speaker.notify('three',[alert]);assert.equal(spoken.length,2);spoken[1].onend(); // Cooldown avoids another slow failure.
  let release;speaker.neuralRetryAt=0;speaker.fetchAudio=()=>new Promise(resolve=>release=resolve);
  speaker.notify('four',[alert]);speaker.enable(false);release(response);await flush();
  assert.equal(played.length,2);assert.equal(spoken.length,2); // Muting cancels in-flight synthesis.
  class BrokenAudio extends Audio{play(){this.onerror();return Promise.reject(new Error('media failed'));}}
  speaker.makeAudio=()=>new BrokenAudio();speaker.fetchAudio=async()=>response;speaker.neuralRetryAt=0;speaker.enable(true);
  speaker.notify('five',[alert]);await flush();assert.equal(spoken.length,3); // Error event + rejected play must not speak twice.
  spoken.at(-1).onend();speaker.enable(false);
  speaker.configure({voice:'neural:unsupported',rate:100,volume:-1});assert.equal(speaker.voice,'auto');assert.equal(speaker.rate,1.15);assert.equal(speaker.volume,0);
  console.log('PASS neural playback, preferences, queue serialization, fallback, cooldown, mute cancellation and error dedup');
}
main().catch(error=>{console.error(error);process.exitCode=1;});

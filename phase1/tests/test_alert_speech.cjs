const assert=require('node:assert/strict');
const AlertSpeech=require('../webui/static/alert_speech.js');
const store=new Map();const storage={getItem:k=>store.get(k),setItem:(k,v)=>store.set(k,v)};
class Utterance{constructor(text){this.text=text;}}
let now=10000;const spoken=[];let cancelled=0;
const synth={getVoices:()=>[{lang:'zh-CN'}],speak:u=>spoken.push(u),cancel:()=>cancelled++};
const speaker=new AlertSpeech({synth,Utterance,storage,now:()=>now});
const alert={behavior_id:'smoking',behavior_name:'吸烟',t_start:2,t_end:4,severity:'medium',reason:'画面可见烟支'};
speaker.seed('bad',{});speaker.notify('bad',[null,42,{},'unconfirmed']);assert.equal(spoken.length,0);
speaker.enable(true);speaker.notify('a',[alert]);
assert.equal(spoken.length,1);assert.match(spoken[0].text,/吸烟/);assert.match(spoken[0].text,/2.0秒/);
speaker.notify('a',[{...alert,t_end:8}]);assert.equal(speaker.queue.length,0);assert.equal(spoken.length,1);
spoken[0].onend();
speaker.seed('b',[alert]);speaker.notify('b',[alert]);assert.equal(spoken.length,1);
speaker.begin('a');speaker.notify('a',[alert],{demo:true,live:true});assert.match(spoken[1].text,/演示告警/);spoken[1].onend();
speaker.notify('a',[{...alert,severity:'high'}]);assert.equal(spoken.length,3);spoken[2].onend();
speaker.notify('many',Array.from({length:30},(_,i)=>({...alert,t_start:i})));
assert.ok(speaker.queue.length<=8);now+=21000;spoken[3].onend();assert.equal(speaker.queue.length,0);
speaker.enable(false);speaker.notify('disabled',[alert]);assert.equal(spoken.length,4);assert.ok(cancelled);
const restored=new AlertSpeech({synth,Utterance,storage});restored.enable(true);restored.notify('a',[alert]);assert.equal(spoken.length,4);restored.enable(false);
const unsupported=new AlertSpeech({synth:null,Utterance:null,storage});assert.equal(unsupported.enable(true),false);
speaker.enable(true);speaker.notify('failure',[alert]);spoken.at(-1).onerror();assert.equal(speaker.enabled,false);
// A live window may expand backwards to include retained context. Identical
// evidence remains the same notification; a new evidence time is a new event.
speaker.enable(true);
const beforeEvidence=spoken.length;
speaker.notify('live-evidence',[{...alert,t_start:0,evidence_time_sec:3.5}],{live:true});
assert.equal(spoken.length,beforeEvidence+1);spoken.at(-1).onend();
speaker.notify('live-evidence',[{...alert,t_start:1,t_end:12,evidence_time_sec:3.5}],{live:true});
assert.equal(spoken.length,beforeEvidence+1);
speaker.notify('live-evidence',[{...alert,t_start:0,evidence_time_sec:9.5}],{live:true});
assert.equal(spoken.length,beforeEvidence+2);spoken.at(-1).onend();
speaker.seed('live-seeded',[{...alert,evidence_time_sec:3.5}],{live:true});
speaker.notify('live-seeded',[{...alert,t_start:0,evidence_time_sec:3.5}],{live:true});
assert.equal(spoken.length,beforeEvidence+2);
// File-video semantics and malformed evidence retain the original start key.
for(const evidence_time_sec of [null,undefined,'3.5',-1,NaN,Infinity]){
  assert.equal(speaker.key('invalid',{...alert,evidence_time_sec},true),speaker.key('invalid',alert));
}
assert.equal(speaker.key('zero',{...alert,evidence_time_sec:0},true),JSON.stringify(['zero','smoking',0,'medium']));
assert.equal(speaker.key('file',{...alert,evidence_time_sec:3.5}),speaker.key('file',alert));
speaker.enable(false);
console.log('Speech: dedup, evolving alerts, restored history, rerun, severity upgrade, bounded queue, stale alerts, mute and errors passed.');

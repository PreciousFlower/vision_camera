const assert=require('node:assert/strict');
const AlertSpeech=require('../webui/static/alert_speech.js');
class Utterance{constructor(text){this.text=text;}}
const spoken=[];let cancelled=0;
const synth={getVoices:()=>[{lang:'zh-CN',name:'Mandarin',voiceURI:'cn'}],speak:u=>spoken.push(u),cancel:()=>cancelled++};
const speaker=new AlertSpeech({synth,Utterance,storage:{getItem:()=>null,setItem:()=>{}},now:()=>1000});
speaker.enable(true);
const pending={source:'workstation',behavior_id:'station_away',behavior_name:'离岗超时',t_start:2,t_end:12,severity:'medium',review_status:'pending'};
speaker.notify('station',[pending],{live:true});
assert.match(spoken[0].text,/待.*复核/);
speaker.notify('station',[],{live:true});
assert.equal(speaker.active,null);assert.equal(cancelled,1);
speaker.notify('other',[{...pending,t_start:20,review_status:'confirmed'}],{live:true});
speaker.notify('station',[{...pending,t_start:30}],{live:true});
assert.equal(speaker.queue.length,1);
speaker.notify('station',[],{live:true});assert.equal(speaker.queue.length,0);
assert.ok(speaker.active);speaker.enable(false);
// Retained historical cards must never be read aloud as fresh violations.
for(const ended of [{review_status:'rejected'},{lifecycle:'withdrawn'},{lifecycle:'resolved'}]){
  const before=spoken.length;speaker.enable(true);
  speaker.notify('retained-'+JSON.stringify(ended),[{...pending,review_status:'confirmed',...ended}],{live:true});
  assert.equal(spoken.length,before,'Withdrawn and resolved alerts are historical, not speakable');
  speaker.enable(false);
}
speaker.enable(true);
const confirmed={...pending,review_status:'confirmed',lifecycle:'active',t_start:50};
speaker.notify('confirmed',[confirmed,{...confirmed,t_start:60}],{live:true});
const playingConfirmed=spoken.at(-1),beforeWithdraw=spoken.length;
speaker.notify('confirmed',[{...confirmed,review_status:'rejected',lifecycle:'withdrawn'},
  {...confirmed,t_start:60,lifecycle:'resolved'}],{live:true});
assert.equal(speaker.active,null,'An explicit withdrawal interrupts a confirmed alert');
assert.equal(speaker.queue.length,0,'Ended queued alerts must be removed even when confirmed');
playingConfirmed.onend();assert.equal(spoken.length,beforeWithdraw);
speaker.notify('confirmed',[{...confirmed,lifecycle:'resolved'},{...confirmed,t_start:60,lifecycle:'resolved'}],{live:true});
assert.equal(spoken.length,beforeWithdraw,'Refreshing retained historical cards does not replay audio');
speaker.enable(false);
console.log('PASS pending speech is explicit; withdrawn/resolved history stays silent and cancels queued or active alerts');

const assert=require('node:assert/strict');
const AlertSpeech=require('../webui/static/alert_speech.js');
class Utterance{constructor(text){this.text=text;}}
const spoken=[];let cancelled=0;
const synth={getVoices:()=>[],speak:u=>spoken.push(u),cancel:()=>cancelled++};
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
console.log('PASS pending speech is explicit; withdrawn workstation alerts cancel pending audio only');

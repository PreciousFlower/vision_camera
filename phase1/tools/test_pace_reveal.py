"""Regression for immediate display, replacing the former artificial pacing.

Runs the actual enqueueWindow function with Node. test_runtime_notice.py
additionally checks real browser cards and visible counters.
"""
from pathlib import Path
import re
import subprocess


def main():
    source = (Path(__file__).resolve().parents[1] / 'webui/static/index.html').read_text(encoding='utf-8')
    match = re.search(r'^function enqueueWindow\([^)]*\)\{(.*?)^\}', source, re.M | re.S)
    if match is None:
        raise RuntimeError('Missing real enqueueWindow function')
    harness = r"""
const shown=[], received=[];
let hints=0;
function renderWindowCard(d,jid){shown.push([d.window_id,jid]);}
function renderPaceHint(){hints++;}
""" + match.group(0) + r"""
for(let batch=0;batch<5;batch++){
  for(let i=0;i<8;i++){
    const id=`${batch}-${i}`;
    received.push([id,'live']);
    enqueueWindow({window_id:id},'live');
  }
  if(shown.length!==received.length) throw Error('Completed windows must show immediately');
}
if(JSON.stringify(shown)!==JSON.stringify(received)) throw Error('A window was lost or reordered');
if(hints!==40) throw Error('Missing display status');
console.log('PASS: all 40 completed windows display immediately, in order, without a timer queue');
"""
    result = subprocess.run(['node', '-e', harness], capture_output=True, text=True)
    print(result.stdout or result.stderr)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())

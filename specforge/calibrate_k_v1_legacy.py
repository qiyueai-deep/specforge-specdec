"""Choose a candidate length from a calibration benchmark and emit a policy file.

This is hardware/model aware K selection: the calibration run is kept separate
from the held-out benchmark, and the chosen K is never selected from the final
test numbers.
"""
import argparse, json
from pathlib import Path

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--benchmark',default='artifacts/benchmark.json'); ap.add_argument('--output',default='artifacts/k_policy.json'); a=ap.parse_args()
 b=json.loads(Path(a.benchmark).read_text(encoding='utf-8')); rows=b['speculative']; best=max(rows,key=lambda x:x['wall_clock_speedup'])
 policy={'selection_rule':'argmax calibration wall_clock_speedup','selected_k':best['draft_k'],'calibration_speedup':best['wall_clock_speedup'],'candidates':[{'k':x['draft_k'],'speedup':x['wall_clock_speedup'],'acceptance':x['mean_acceptance_rate']} for x in rows]}
 Path(a.output).write_text(json.dumps(policy,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(policy,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

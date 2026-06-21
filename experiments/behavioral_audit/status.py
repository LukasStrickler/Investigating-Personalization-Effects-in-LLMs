import csv, json, time, os
from pathlib import Path                                                                                                                                                                                         
from collections import Counter                                                                                                                                                                                

for q in ['q1', 'q2']:
    d = Path(f'logs/behavioral-audit-full002-{q}-stage1')
    if not d.exists(): print(f'stage1 {q}: not started'); continue
    csvs = sorted(d.glob('*.csv'), key=lambda p: p.stat().st_mtime)
    if not csvs: continue
    p = csvs[-1]
    age = time.time() - os.path.getmtime(p)
    with open(p) as f:
        rows = list(csv.DictReader(f))
    MODELS = ['grok-4.3_paid', 'glm-5.2_paid']
    statuses = Counter()
    per_model = {m: Counter() for m in MODELS}
    for r in rows:
        for model in MODELS:
            v = r.get(model, '')
            if not v: statuses['pending'] += 1; per_model[model]['pending'] += 1; continue
            try:
                s = json.loads(v).get('status', 'unknown')
                statuses[s] += 1
                per_model[model][s] += 1
            except: pass
    total = sum(statuses.values())
    success = statuses.get('success', 0)
    pct = 100 * success / total if total else 0
    model_parts = '  '.join(
        f"{m.split('_')[0]}: {per_model[m].get('success',0)}/{sum(per_model[m].values())} ({100*per_model[m].get('success',0)/sum(per_model[m].values()):.0f}%)"
        for m in MODELS if sum(per_model[m].values()) > 0
    )
    print(f'stage1 {q}: {success}/{total} ({pct:.0f}%)  [{model_parts}]  last write {age:.0f}s ago')

for q in ['q1', 'q2']:
    p = Path(f'logs/judges/behavioral-audit/behavioral-audit-full002-{q}-stage2.judgments.csv')
    if not p.exists(): print(f'stage2 {q}: not started'); continue
    age = time.time() - os.path.getmtime(p)
    with open(p) as f:
        rows = list(csv.DictReader(f))
    current_hash = Counter(r['judge_config_hash'] for r in rows).most_common(1)[0][0]
    # deduplicate: latest per (subject_id, subject_model_alias)
    by_subject = {}
    for r in rows:
        if r['judge_config_hash'] != current_hash: continue
        key = (r['subject_id'], r['subject_model_alias'])
        if key not in by_subject or r['completed_at'] > by_subject[key]['completed_at']:
            by_subject[key] = r
    current = list(by_subject.values())
    success = sum(1 for r in current if r['status'] == 'success')
    total = len(current)
    pct = 100 * success / total if total else 0
    fail_status = Counter(r['status'] for r in current if r['status'] != 'success')
    fail_reasons = Counter(r['error_message'][:60] for r in current if r['status'] != 'success')
    fail_str = '  '.join(f'{s}:{n}' for s, n in fail_status.items())
    reason_str = '  '.join(f'{n}x "{reason}"' for reason, n in fail_reasons.most_common(3))
    print(f'stage2 {q}: {success}/{total} ({pct:.0f}%)  fails: {fail_str}  last write {age:.0f}s ago')
    if reason_str:
        print(f'         reasons: {reason_str}')
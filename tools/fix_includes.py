"""Batch update #include paths after region/ restructuring."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPLACEMENTS = {
    '"region/basal_ganglia.h"': '"region/subcortical/basal_ganglia.h"',
    '"region/thalamic_relay.h"': '"region/subcortical/thalamic_relay.h"',
    '"region/cerebellum.h"': '"region/subcortical/cerebellum.h"',
    '"region/hippocampus.h"': '"region/limbic/hippocampus.h"',
    '"region/amygdala.h"': '"region/limbic/amygdala.h"',
    '"region/vta_da.h"': '"region/neuromod/vta_da.h"',
    '"region/lc_ne.h"': '"region/neuromod/lc_ne.h"',
    '"region/drn_5ht.h"': '"region/neuromod/drn_5ht.h"',
    '"region/nbm_ach.h"': '"region/neuromod/nbm_ach.h"',
}

changed = 0
for search_dir in ['src', 'tests']:
    for dirpath, _, filenames in os.walk(os.path.join(ROOT, search_dir)):
        for fn in filenames:
            if not fn.endswith(('.h', '.cpp')):
                continue
            fp = os.path.join(dirpath, fn)
            with open(fp, 'r', encoding='utf-8') as f:
                content = f.read()
            new = content
            for old, repl in REPLACEMENTS.items():
                new = new.replace(old, repl)
            if new != content:
                with open(fp, 'w', encoding='utf-8') as f:
                    f.write(new)
                changed += 1
                print(f'  Updated: {os.path.relpath(fp, ROOT)}')

print(f'\nTotal files updated: {changed}')

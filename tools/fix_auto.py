"""Replace 'auto x = y.step_and_compute(...)' with 'const auto& x = y.step_and_compute(...)' in all .cpp files under src/"""
import os, re

root = r'k:\Cherry\LLM\AGI\agi3\src'
pattern = re.compile(r'(\s+)auto (\w+) = (\w+[\._]*\w*)\.step_and_compute\(')
replacement = r'\1const auto& \2 = \3.step_and_compute('

total = 0
for dp, _, fns in os.walk(root):
    for f in fns:
        if not f.endswith('.cpp'):
            continue
        path = os.path.join(dp, f)
        with open(path, 'r', encoding='utf-8') as fh:
            content = fh.read()
        new_content, n = pattern.subn(replacement, content)
        if n > 0:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(new_content)
            print(f'  {f}: {n} replacements')
            total += n

print(f'Total: {total} replacements')

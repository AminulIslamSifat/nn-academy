#!/usr/bin/env python3
"""Fix multi-line code: strings in MDX quiz sections."""
from pathlib import Path

CONTENT_DIR = Path("/home/sifat/hdd/projects/numpy-nn-academy/src/content/chapters")
BSLASH_N = chr(92) + "n"  # literal backslash-n

fixed = 0
for f in sorted(CONTENT_DIR.glob("*.mdx")):
    lines = f.read_text().split("\n")
    new_lines = []
    i = 0
    changed = False
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        has_code_multiline = ('code:"' in stripped or 'code: "' in stripped) and not (stripped.rstrip().endswith('",') or stripped.rstrip().endswith('",}') or stripped.rstrip().endswith('"}') or stripped.rstrip().endswith('"'))
        # Also check if code: appears mid-line and the string doesn't close on same line
        if not has_code_multiline and 'code:"' in line:
            idx = line.index('code:"') + 6
            rest = line[idx:]
            # Count quotes — if odd number, string doesn't close on this line
            if rest.count('"') % 2 == 1:
                has_code_multiline = True
        if has_code_multiline:
            code_lines = [line.rstrip()]
            j = i + 1
            while j < len(lines):
                code_lines.append(lines[j])
                if lines[j].rstrip().endswith('",') or lines[j].rstrip().endswith('"'):
                    break
                j += 1
            full = "\n".join(code_lines)
            first_q = full.index('"')
            last_q = full.rindex('"')
            code_val = full[first_q + 1 : last_q]
            prefix = full[: first_q + 1]
            suffix = full[last_q:]
            escaped = code_val.replace("\n", BSLASH_N)
            new_lines.append(prefix + escaped + suffix)
            i = j + 1
            changed = True
        else:
            new_lines.append(line)
            i += 1

    if changed:
        f.write_text("\n".join(new_lines))
        fixed += 1
        print(f"  Fixed: {f.name}")

print(f"\nFixed {fixed} files")

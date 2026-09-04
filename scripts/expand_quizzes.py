#!/usr/bin/env python3
"""Convert compact single-line quiz format to multi-line MDX-compatible format."""
import re
from pathlib import Path

CONTENT_DIR = Path("/home/sifat/hdd/projects/numpy-nn-academy/src/content/chapters")

def expand_quiz(content):
    """Find compact <Quiz .../> and expand to multi-line format."""
    # Match <Quiz chapterSlug="..." questions={[...]} />
    pattern = r'<Quiz\s+chapterSlug="([^"]+)"\s+questions=\{\[(.*?)\]\}\s*/>'
    
    def replacer(match):
        slug = match.group(1)
        questions_raw = match.group(2)
        
        # Parse individual question objects
        # Split by }, { boundaries
        questions = []
        depth = 0
        current = ""
        for ch in questions_raw:
            if ch == '{':
                depth += 1
                current += ch
            elif ch == '}':
                depth -= 1
                current += ch
                if depth == 0:
                    questions.append(current.strip())
                    current = ""
            else:
                current += ch
        
        # Format each question as multi-line
        formatted_questions = []
        for q in questions:
            # Remove outer braces
            inner = q.strip()
            if inner.startswith('{'):
                inner = inner[1:]
            if inner.endswith('}'):
                inner = inner[:-1]
            
            # Parse key:value pairs
            # Handle arrays in options carefully
            pairs = {}
            i = 0
            while i < len(inner):
                # Skip whitespace and commas
                while i < len(inner) and inner[i] in ' ,\n':
                    i += 1
                if i >= len(inner):
                    break
                
                # Read key
                key_match = re.match(r'(\w+)\s*:', inner[i:])
                if not key_match:
                    i += 1
                    continue
                key = key_match.group(1)
                i += key_match.end()
                
                # Skip whitespace
                while i < len(inner) and inner[i] in ' \n':
                    i += 1
                
                # Read value
                if i < len(inner) and inner[i] == '[':
                    # Array value — find matching ]
                    bracket_depth = 0
                    start = i
                    while i < len(inner):
                        if inner[i] == '[':
                            bracket_depth += 1
                        elif inner[i] == ']':
                            bracket_depth -= 1
                            if bracket_depth == 0:
                                i += 1
                                break
                        i += 1
                    pairs[key] = inner[start:i]
                elif i < len(inner) and inner[i] == '"':
                    # String value — find closing quote (handle escaped quotes)
                    start = i
                    i += 1
                    while i < len(inner):
                        if inner[i] == chr(92) and i + 1 < len(inner):
                            i += 2
                        elif inner[i] == '"':
                            i += 1
                            break
                        else:
                            i += 1
                    pairs[key] = inner[start:i]
                else:
                    # Number, boolean, etc — read until comma or end
                    start = i
                    while i < len(inner) and inner[i] not in ',}':
                        i += 1
                    pairs[key] = inner[start:i].strip()
            
            # Build multi-line question
            lines = ["    {"]
            key_order = ["id", "type", "prompt", "code", "options", "correctIndex", "explanation", "randomize"]
            for key in key_order:
                if key in pairs:
                    val = pairs[key]
                    lines.append(f'      {key}: {val},')
            # Add any remaining keys
            for key, val in pairs.items():
                if key not in key_order:
                    lines.append(f'      {key}: {val},')
            lines.append("    }")
            formatted_questions.append("\n".join(lines))
        
        # Build the full multi-line Quiz component
        result = f'<Quiz\n  chapterSlug="{slug}"\n  questions={{[\n'
        result += ",\n".join(formatted_questions)
        result += "\n  ]}\n/>"
        return result
    
    return re.sub(pattern, replacer, content, flags=re.DOTALL)

fixed = 0
for f in sorted(CONTENT_DIR.glob("*.mdx")):
    content = f.read_text()
    new_content = expand_quiz(content)
    if new_content != content:
        f.write_text(new_content)
        fixed += 1
        print(f"  Expanded: {f.name}")

print(f"\nExpanded {fixed} files")

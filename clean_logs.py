#!/usr/bin/env python3
import re
import sys

ANSI_PATTERN = re.compile(r'\x1b\[[0-9;]*m')

def clean_line(line: str) -> str:
    return ANSI_PATTERN.sub('', line)

def clean_file(input_path: str, output_path: str = None):
    if output_path is None:
        output_path = input_path.replace('.txt', '_clean.txt')
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(clean_line(line))
    
    print(f"Cleaned {len(lines)} lines → {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python clean_logs.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    clean_file(input_file, output_file)
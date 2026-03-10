#!/usr/bin/env python3
import site
import os
import subprocess
import glob

print("Starting ctranslate2 library patching...")

libs_found = []
paths = site.getsitepackages()

for p in paths:
    for lib in glob.glob(os.path.join(p, '**', '*.so*'), recursive=True):
        if 'ctranslate2' in lib and os.path.isfile(lib):
            libs_found.append(lib)

print(f"Found {len(libs_found)} ctranslate2 libraries:")
for lib in libs_found:
    print(f"  - {lib}")

for lib in libs_found:
    print(f"Patching: {lib}")
    result = subprocess.run(['patchelf', '--clear-execstack', lib], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ Success")
    else:
        print(f"  ✗ Failed: {result.stderr}")

print("Patching complete!")

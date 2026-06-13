import os
import glob
import re

files = glob.glob('templates/admin_*.html')
files = [f for f in files if 'dashboard' not in f and 'login' not in f]

for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Update the main container
    content = content.replace('class="w-full"', 'class="max-w-[1400px] mx-auto pb-10"')
    content = content.replace('class="max-w-7xl mx-auto"', 'class="max-w-[1400px] mx-auto pb-10"')
    
    # 2. Update Header sizes
    content = content.replace('text-2xl font-semibold text-admin-text mb-1', 'text-[32px] font-semibold text-admin-text tracking-tight mb-1')
    content = content.replace('text-sm text-admin-textMuted font-medium', 'text-[15px] text-admin-textMuted')

    # 3. Update table headers to match new small styling (text-[11px])
    content = content.replace('text-xs font-bold text-admin-textMuted uppercase', 'text-[11px] font-bold text-admin-textMuted uppercase')

    # 4. Make card headers match Dashboard layout (text-lg font-semibold to be standard)
    # The stats grid usually has `rounded-lg`. Let's ensure it doesn't have redundant borders since .admin-card handles it
    content = content.replace('border border-admin-border rounded-lg', '') # remove redundant classes from stats cards since .admin-card handles it.
    content = content.replace('class="admin-card ', 'class="admin-card ')
    
    # In table wrappers, remove redundant rounded-lg since admin-card handles border-radius
    content = content.replace('admin-card overflow-hidden', 'admin-card !p-0 overflow-hidden flex flex-col')
    content = content.replace('admin-card p-0 overflow-hidden', 'admin-card !p-0 overflow-hidden flex flex-col')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"Updated layout classes for {filepath}")

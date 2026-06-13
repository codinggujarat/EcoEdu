import os
import re

TEMPLATES_DIR = 'templates'

INPUT_CLASS_PATTERN = re.compile(r'class="w-full h-11[^"]*border[^"]*"')
TEXTAREA_CLASS_PATTERN = re.compile(r'class="w-full px-4 py-3[^"]*resize-none"')

# Colors to remove or replace
COLOR_REPLACEMENTS = [
    (r'bg-blue-50', 'bg-[#F4F4EF]'),
    (r'text-blue-600', 'text-[#181D00]'),
    (r'bg-purple-50', 'bg-[#F4F4EF]'),
    (r'text-purple-600', 'text-[#181D00]'),
    (r'bg-green-50', 'bg-gray-100'),
    (r'text-green-600', 'text-[#181D00]'),
    (r'bg-orange-50', 'bg-[#F4F4EF]'),
    (r'text-orange-600', 'text-[#181D00]'),
    (r'bg-red-50', 'bg-gray-100'),
    (r'text-red-600', 'text-[#181D00]'),
    (r'border-t-blue-500', 'border-t-[#181D00]'),
    (r'border-t-purple-500', 'border-t-[#181D00]'),
    (r'border-t-amber-500', 'border-t-[#181D00]'),
    (r'border-t-4 border-t-[#181D00]', ''), # Just remove colored top borders
    (r'bg-green-100 text-green-800', 'bg-gray-100 text-[#181D00]'),
    (r'bg-blue-100 text-blue-800', 'bg-gray-100 text-[#181D00]'),
    (r'bg-purple-100 text-purple-800', 'bg-gray-100 text-[#181D00]'),
    (r'bg-yellow-100 text-yellow-800', 'bg-gray-100 text-[#181D00]'),
]

for filename in os.listdir(TEMPLATES_DIR):
    if not filename.startswith('admin_') or not filename.endswith('.html'):
        continue
        
    filepath = os.path.join(TEMPLATES_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    original_content = content
    
    # 1. Standardize form inputs
    content = INPUT_CLASS_PATTERN.sub('class="admin-input"', content)
    content = TEXTAREA_CLASS_PATTERN.sub('class="admin-textarea"', content)
    
    # 2. Replace colors
    for old, new in COLOR_REPLACEMENTS:
        content = re.sub(old, new, content)
        
    # 3. Form spacing - change p-6 in form cards to p-8
    # Search for premium-card in form pages
    if 'form' in content.lower() and filename != 'admin_login.html':
        content = content.replace('premium-card p-6', 'premium-card p-8')
        content = content.replace('premium-card p-5', 'premium-card p-8')
        content = content.replace('gap-6', 'gap-8')
        content = content.replace('gap-4', 'gap-8')
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {filename}")

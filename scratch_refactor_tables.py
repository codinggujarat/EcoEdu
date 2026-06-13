import os
import re

TEMPLATES_DIR = 'templates'

for filename in os.listdir(TEMPLATES_DIR):
    if not filename.startswith('admin_') or not filename.endswith('.html'):
        continue
        
    filepath = os.path.join(TEMPLATES_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    original_content = content
    
    # Increase padding in table toolbars and forms
    content = content.replace('p-4 border-b', 'p-6 border-b')
    content = content.replace('p-5 border-b', 'p-6 border-b')
    
    # Search inputs standardization
    content = content.replace('w-full h-9 pl-9 pr-4 text-sm bg-white border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-100 focus:border-blue-400 outline-none transition-all placeholder:text-gray-400', 'admin-input pl-10')
    content = content.replace('w-full h-10 pl-9 pr-4 text-sm bg-white border border-gray-200 rounded-lg focus:ring-2 focus:ring-green-100 focus:border-green-600 outline-none transition-all placeholder:text-gray-400', 'admin-input pl-10')

    # Remove any remaining blue rings
    content = content.replace('focus:ring-blue-100', 'focus:ring-gray-100')
    content = content.replace('focus:border-blue-400', 'focus:border-gray-900')
    
    # Any residual accent colors inside tables
    content = content.replace('bg-green-100 text-green-800', 'bg-gray-100 text-[#181D00]')
    content = content.replace('bg-blue-100 text-blue-800', 'bg-gray-100 text-[#181D00]')
    content = content.replace('bg-purple-100 text-purple-800', 'bg-gray-100 text-[#181D00]')
    content = content.replace('bg-yellow-100 text-yellow-800', 'bg-gray-100 text-[#181D00]')
    content = content.replace('bg-orange-100 text-orange-800', 'bg-gray-100 text-[#181D00]')
    content = content.replace('bg-red-100 text-red-800', 'bg-gray-100 text-[#181D00]')

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {filename}")

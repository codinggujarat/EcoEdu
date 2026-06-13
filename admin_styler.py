import os
import re

files = [
    'admin_students.html',
    'admin_teacher.html',
    'admin_add_challenge.html',
    'admin_achievements.html',
    'admin_add_achievement.html',
    'admin_eco_tips.html',
    'admin_levels.html',
    'admin_puzzles.html'
]

css_block = '''<style>
 /* Professional DataTables Override */
 .dataTables_wrapper .dataTables_length,
 .dataTables_wrapper .dataTables_filter {
 padding: 1rem 1.5rem;
 color: #181D00;
 font-size: 0.875rem;
 font-weight: 500;
 }

 .dataTables_wrapper .dataTables_filter input {
 background: #FFFFFF;
 border: 1px solid rgba(24, 29, 0, 0.15);
 border-radius: 6px;
 color: #181D00;
 padding: 6px 12px;
 margin-left: 8px;
 outline: none;
 }
 .dataTables_wrapper .dataTables_filter input:focus {
 border-color: #181D00;
 }

 .dataTables_wrapper .dataTables_info {
 padding: 1rem 1.5rem;
 color: rgba(24,29,0,0.6);
 font-size: 0.875rem;
 }

 .dataTables_wrapper .dataTables_paginate {
 padding: 1rem 1.5rem;
 font-size: 0.875rem;
 }

 .dataTables_wrapper .paginate_button {
 color: rgba(24,29,0,0.6) !important;
 padding: 4px 10px !important;
 margin: 0 2px;
 border-radius: 6px;
 border: 1px solid transparent !important;
 cursor: pointer;
 }
 .dataTables_wrapper .paginate_button:hover {
 background: rgba(24,29,0,0.05) !important;
 color: #181D00 !important;
 border: 1px solid transparent !important;
 }

 .dataTables_wrapper .paginate_button.current, .dataTables_wrapper .paginate_button.current:hover {
 color: #FFFFFF !important;
 background: #181D00 !important;
 border: 1px solid #181D00 !important;
 font-weight: 500;
 }
</style>'''

for filename in files:
    filepath = os.path.join('templates', filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the old style block
        content = re.sub(r'<style>.*?Minimalist DataTables.*?</style>', css_block, content, flags=re.DOTALL)
        content = re.sub(r'<style>.*?DataTables Minimalist Override.*?</style>', css_block, content, flags=re.DOTALL)
        
        # Replace the tbody background
        content = content.replace('<tbody class="divide-y divide-admin-border">', '<tbody class="divide-y divide-admin-border bg-[#F7F7F2]">')
        
        # Replace generic dark colors in charts
        content = content.replace("'#2A2E35'", "'rgba(24,29,0,0.1)'")
        content = content.replace("'#A1A1AA'", "'rgba(24,29,0,0.6)'")
        content = content.replace("'#E6E8EB'", "'#181D00'")
        
        # Header backgrounds for tables
        content = content.replace('bg-[#E5E7EB]/50', 'bg-white border-b border-admin-border')
        content = content.replace('bg-[#E5E7EB]', 'bg-white')
        
        # Make buttons use white background or standard border
        content = content.replace('bg-[#FFFFFF]', 'bg-white')
        
        # Any remaining bg-[#171A1F] logic
        content = content.replace('bg-[#171A1F]', 'admin-card')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Polished {filename}')

import os
import re
import glob

files = glob.glob('templates/admin*.html')

css_block = '''<style>
 /* Professional DataTables Override */
 .dataTables_wrapper .dataTables_length,
 .dataTables_wrapper .dataTables_filter {
 padding: 1rem 1.5rem;
 color: #111111;
 font-size: 0.875rem;
 font-weight: 500;
 }

 .dataTables_wrapper .dataTables_filter input {
 background: #FFFFFF;
 border: 1px solid rgba(0,0,0,0.12);
 border-radius: 6px;
 color: #111111;
 padding: 6px 12px;
 margin-left: 8px;
 outline: none;
 transition: none;
 }
 .dataTables_wrapper .dataTables_filter input:focus {
 border-color: #2A4A30;
 box-shadow: 0 0 0 1px #2A4A30;
 }

 .dataTables_wrapper .dataTables_info {
 padding: 1rem 1.5rem;
 color: #666666;
 font-size: 0.875rem;
 }

 .dataTables_wrapper .dataTables_paginate {
 padding: 1rem 1.5rem;
 font-size: 0.875rem;
 }

 .dataTables_wrapper .paginate_button {
 color: #666666 !important;
 padding: 4px 10px !important;
 margin: 0 2px;
 border-radius: 6px;
 border: 1px solid transparent !important;
 cursor: pointer;
 transition: none;
 }
 .dataTables_wrapper .paginate_button:hover {
 background: rgba(0,0,0,0.05) !important;
 color: #111111 !important;
 border: 1px solid transparent !important;
 }

 .dataTables_wrapper .paginate_button.current, .dataTables_wrapper .paginate_button.current:hover {
 color: #FFFFFF !important;
 background: #2A4A30 !important;
 border: 1px solid #2A4A30 !important;
 font-weight: 500;
 }
</style>'''

for filepath in files:
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the DataTables CSS block
        content = re.sub(r'<style>.*?Professional DataTables Override.*?</style>', css_block, content, flags=re.DOTALL)
        
        # Replace old table alt row colors
        content = content.replace('bg-[#F7F7F2]', 'bg-admin-tableAlt')
        
        # Replace text classes
        content = content.replace('text-admin-textMuted', 'text-admin-textMuted') # Already using class
        content = content.replace('text-[#181D00]', 'text-admin-text')
        content = content.replace('text-[#707070]', 'text-admin-textMuted')
        content = content.replace('text-gray-600', 'text-admin-textMuted')
        
        # Replace chart generic dark colors with new ones
        content = content.replace("'rgba(24,29,0,0.1)'", "'rgba(0,0,0,0.08)'")
        content = content.replace("'rgba(24,29,0,0.6)'", "'#888888'")
        content = content.replace("'#181D00'", "'#2A4A30'")
        
        # Replace main body bg classes
        content = content.replace('bg-white', 'bg-admin-surface')
        
        # Replace card borders
        content = content.replace('border-admin-border', 'border-admin-border')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Polished {filepath}')

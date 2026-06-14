import os
import re

template_dir = 'c:\\GITHUB\\EcoEdu\\templates'

sidebar_files = ['teacher_dashboard_sidebar.html', 'student_dashboard_sidebar.html']
exclude_classes = ['sidebar-link', 'nav-link', 'footer-col', 'nav-row', 'text-part', 'active-indicator']

for filename in os.listdir(template_dir):
    if not filename.endswith('.html'):
        continue
    if filename in sidebar_files:
        continue # skip sidebar files entirely
        
    filepath = os.path.join(template_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 1. Remove all rules containing :hover from CSS blocks.
    # regex: match a selector containing :hover, followed by { ... }
    # but don't match if it's .sidebar-link:hover etc.
    def hover_replacer(match):
        selector = match.group(1)
        body = match.group(2)
        
        # If it's a sidebar/nav class, keep it
        for ex in exclude_classes:
            if ex in selector:
                return match.group(0)
                
        # Otherwise remove it completely
        return ""

    # This regex looks for selector containing :hover and its body
    # It handles simple {} blocks but might fail on nested media queries.
    # Fortunately most of our hovers are not inside media queries.
    content = re.sub(r'([^\{\}]*?:hover[^\{\}]*?)\{([^}]*)\}', hover_replacer, content, flags=re.MULTILINE | re.DOTALL)

    # 2. Also strip "transition:" and "animation:" from all remaining inline or internal CSS in these files
    content = re.sub(r'transition:\s*[^;\}]+;?', '', content)
    content = re.sub(r'animation:\s*[^;\}]+;?', '', content)
    
    # 3. Strip GSAP and Swup transition logic from dashboard_layout.html
    if filename == 'dashboard_layout.html':
        # Remove Swup transition classes
        content = re.sub(r'\.transition-fade\s*\{[^}]*\}', '', content, flags=re.DOTALL)
        content = re.sub(r'html\.is-animating\s*\.transition-fade\s*\{[^}]*\}', '', content, flags=re.DOTALL)
        
        # Empty initGSAPAnimations
        gsap_regex = r'(function initGSAPAnimations\(\) \{).*?(\n\})'
        content = re.sub(r'function initGSAPAnimations\(\)\s*\{[\s\S]*?^\}', 'function initGSAPAnimations() {\n    // Animations removed per request\n}', content, flags=re.MULTILINE)
        
        # Remove GSAP script imports
        content = re.sub(r'<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap.*?</script>\n', '', content)

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Cleaned {filename}")

print("Done stripping animations!")

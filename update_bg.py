import os
import re

def update_templates():
    templates_dir = "templates"
    
    # We will look for CSS rules that define sections vs cards.
    # User said: main page background -> #F3F4E5
    # Card backgrounds -> #FFFFFF
    
    # Files to process (user-side only, NO admin)
    user_files = [
        "student_dashboard.html",
        "teacher_dashboard.html",
        "challenges.html",
        "leaderboard.html",
        "puzzles.html",
        "profile.html",
        "teacher_profiles.html",
        "teacher_student_profile.html",
        "teacher_verify.html",
        "login.html",
        "register.html",
        "reset_password.html",
        "verify_otp.html",
    ]
    
    for filename in user_files:
        filepath = os.path.join(templates_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        original_content = content
        
        # 1. Replace section backgrounds with #F3F4E5
        # Usually defined like .premium-section { background: #F7F7F2; ... }
        # or .ch-section { background: #F7F7F2; ... }
        content = re.sub(r'(\.(?:premium|ch|td|tp|pf|tv)-section\s*\{\s*background:\s*)#[a-fA-F0-9]{3,6}', r'\g<1>#F3F4E5', content)
        content = re.sub(r'(\.(?:premium|ch|td|tp|pf|tv)-section-alt\s*\{\s*background:\s*)#[a-fA-F0-9]{3,6}', r'\g<1>#F3F4E5', content)
        
        # 2. Replace card backgrounds with #FFFFFF
        # Usually defined like .ch-card { background: #F7F7F2; ... }
        # or .edit-auth-card { background: #F7F7F2; ... }
        content = re.sub(r'(\.(?:pz|ch|td|tv)-card\s*\{\s*[^}]*background:\s*)#[a-fA-F0-9]{3,6}', r'\g<1>#FFFFFF', content)
        content = re.sub(r'(\.edit-auth-card\s*\{\s*background:\s*)#[a-fA-F0-9]{3,6}', r'\g<1>#FFFFFF', content)
        content = re.sub(r'(\.carousel-item\s*\{[^\}]*background:\s*)#[a-fA-F0-9]{3,6}', r'\g<1>#FFFFFF', content)
        
        # 3. Replace inline styles for cards (like in profile.html)
        content = re.sub(r'(background:\s*)#F7F7F2(;[^>]*border:\s*1px solid rgba)', r'\g<1>#FFFFFF\2', content)
        
        # 4. Replace any stray #FFFFFF that might have been acting as page background (if any)
        # We will manually change body { background: #FFFFFF; } if it exists, to #F3F4E5
        content = re.sub(r'(body\s*\{\s*background(?:-color)?:\s*)#FFFFFF', r'\g<1>#F3F4E5', content)
        
        # 5. Fix edit-input backgrounds to match card or be slightly different? Keep as #FFFFFF
        content = re.sub(r'(\.edit-input\s*\{[^\}]*background:\s*)#[a-fA-F0-9]{3,6}', r'\g<1>#FFFFFF', content)
        
        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Updated {filename}")

if __name__ == "__main__":
    update_templates()

import werkzeug.security
import hmac

# Monkey patch
if not hasattr(werkzeug.security, 'safe_str_cmp'):
    werkzeug.security.safe_str_cmp = lambda a, b: hmac.compare_digest(a.encode('utf-8'), b.encode('utf-8'))

from app import app, db, User

with app.app_context():
    # 1. Update NULLs to 'default.png'
    # 2. Update empty strings to 'default.png'
    # 3. Update literal "None" string to 'default.png'
    
    users = User.query.all()
    count = 0
    for u in users:
        changed = False
        original = u.profile_pic
        
        if u.profile_pic is None:
            u.profile_pic = 'default.png'
            changed = True
        elif str(u.profile_pic).strip() == "":
            u.profile_pic = 'default.png'
            changed = True
        elif str(u.profile_pic).strip().lower() == "none":
            u.profile_pic = 'default.png'
            changed = True
            
        if changed:
            print(f"Fixed user {u.name}: '{original}' -> 'default.png'")
            count += 1
            
    if count > 0:
        db.session.commit()
        print(f"✅ Successfully updated {count} user profile pictures to default.png")
    else:
        print("No users needed fixing.")

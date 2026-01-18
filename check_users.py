import werkzeug.security
import hmac

# Monkey patch safe_str_cmp for compatibility with older libraries
if not hasattr(werkzeug.security, 'safe_str_cmp'):
    werkzeug.security.safe_str_cmp = lambda a, b: hmac.compare_digest(a.encode('utf-8'), b.encode('utf-8'))

from app import app, db, User

with app.app_context():
    users = User.query.all()
    print(f"{'Name':<20} | {'Role':<10} | {'Profile Pic':<30} | {'Type'}")
    print("-" * 80)
    for u in users:
        print(f"{u.name:<20} | {u.role:<10} | {str(u.profile_pic):<30} | {type(u.profile_pic)}")


from app import app, db, User, Achievement
from sqlalchemy import text

def update_db():
    with app.app_context():
        print("Checking database schema...")
        engine = db.engine
        inspector = db.inspect(engine)
        
        # 1. Create Certificate Table (db.create_all will handle this if table is missing)
        db.create_all()
        print("Ensured all tables exist (including Certificate).")
        
        # 2. Add 'challenges_completed' to User if missing
        user_cols = [c['name'] for c in inspector.get_columns('user')]
        if 'challenges_completed' not in user_cols:
            print("Adding 'challenges_completed' to User table...")
            with engine.connect() as con:
                con.execute(text('ALTER TABLE user ADD COLUMN challenges_completed INTEGER DEFAULT 0'))
        
        # 3. Add 'challenges_required' to Achievement if missing
        ach_cols = [c['name'] for c in inspector.get_columns('achievement')]
        if 'challenges_required' not in ach_cols:
            print("Adding 'challenges_required' to Achievement table...")
            with engine.connect() as con:
                con.execute(text('ALTER TABLE achievement ADD COLUMN challenges_required INTEGER DEFAULT 0'))
        
        print("Database update complete.")

if __name__ == "__main__":
    update_db()

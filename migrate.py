import sqlite3
conn = sqlite3.connect('eco_education.db')
cursor = conn.cursor()
try:
    cursor.execute('ALTER TABLE challenge_completion ADD COLUMN status VARCHAR(20) DEFAULT "pending"')
    cursor.execute('UPDATE challenge_completion SET status = "approved" WHERE verified = 1')
    conn.commit()
    print("Migration successful")
except Exception as e:
    print("Migration error:", e)
finally:
    conn.close()

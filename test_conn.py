import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
engine = create_engine(os.environ['DATABASE_URL'])

with engine.connect() as conn:
    print('Tables Accessible:')
    tables = ['user', 'challenge', 'achievement', 'puzzle', 'level', 'certificate', 'eco_tip']
    for t in tables:
        count = conn.execute(text(f'SELECT COUNT(*) FROM "{t}"')).scalar()
        print(f'{t}: {count}')

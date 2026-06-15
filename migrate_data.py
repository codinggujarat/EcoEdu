import os
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv

def run_migration():
    # Load .env so it picks up the external PostgreSQL URL
    load_dotenv()
    
    # Source SQLite database
    source_url = "sqlite:///eco_education.db"
    
    # Destination PostgreSQL database (from env)
    dest_url = os.environ.get('DATABASE_URL')
    if dest_url and dest_url.startswith('postgres://'):
        dest_url = dest_url.replace('postgres://', 'postgresql://', 1)

    print(f"Connecting to Source: {source_url}")
    # Mask password for printing
    safe_dest = dest_url
    if "@" in dest_url:
        parts = dest_url.split("@")
        creds = parts[0].split(":")
        if len(creds) >= 3:
            safe_dest = f"{creds[0]}:{creds[1]}:***@{parts[1]}"
    print(f"Connecting to Destination: {safe_dest}")

    try:
        source_engine = create_engine(source_url)
        dest_engine = create_engine(dest_url)
        
        # Test connections
        with source_engine.connect() as s, dest_engine.connect() as d:
            pass
    except Exception as e:
        print(f"CRITICAL ERROR: Could not connect to databases. {e}")
        print("Note: If the destination is an internal Render URL (dpg-...), you MUST run this script in the Render Shell!")
        return

    source_meta = MetaData()
    source_meta.reflect(bind=source_engine)

    dest_meta = MetaData()
    dest_meta.reflect(bind=dest_engine)

    # Ordered to respect foreign keys
    tables_to_migrate = [
        'user',
        'challenge',
        'achievement',
        'puzzle',
        'level',
        'certificate',
        'eco_tip',
        'user_achievement',
        'challenge_completion',
        'puzzle_completion',
        'otp_verification',
        'password_reset_token'
    ]

    with source_engine.connect() as src_conn:
        with dest_engine.connect() as dest_conn:
            for table_name in tables_to_migrate:
                if table_name not in source_meta.tables or table_name not in dest_meta.tables:
                    print(f"Skipping {table_name}: Table missing.")
                    continue
                    
                src_table = source_meta.tables[table_name]
                dest_table = dest_meta.tables[table_name]
                
                rows = src_conn.execute(src_table.select()).mappings().all()
                
                success_count = 0
                skip_count = 0
                
                for row in rows:
                    try:
                        dest_conn.execute(dest_table.insert().values(**dict(row)))
                        dest_conn.commit()
                        success_count += 1
                    except IntegrityError:
                        # Row already exists (Primary key or Unique constraint violation)
                        dest_conn.rollback()
                        skip_count += 1
                    except Exception as e:
                        dest_conn.rollback()
                        print(f"Error migrating row in {table_name}: {e}")
                
                print(f"Table {table_name:<20} | Migrated: {success_count:<4} | Skipped (Existing): {skip_count:<4} | Total Source: {len(rows)}")

            # Update PostgreSQL sequences so new inserts don't fail with PK violations
            print("\nUpdating PostgreSQL auto-increment sequences...")
            for table_name in tables_to_migrate:
                try:
                    query = text(f"SELECT setval(pg_get_serial_sequence('{table_name}', 'id'), COALESCE((SELECT MAX(id) FROM {table_name}), 1), max(id) IS NOT null) FROM {table_name};")
                    dest_conn.execute(query)
                    dest_conn.commit()
                except Exception:
                    dest_conn.rollback() # Not all tables have 'id' sequence (e.g. association tables)
            
            # Print final verification counts
            print("\n--- FINAL DESTINATION ROW COUNTS ---")
            for table_name in tables_to_migrate:
                if table_name in dest_meta.tables:
                    count = dest_conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
                    print(f"{table_name:<20}: {count}")

    print("\nMigration Script Finished Successfully!")

if __name__ == "__main__":
    run_migration()

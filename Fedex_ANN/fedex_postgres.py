# ============================================================
# POSTGRESQL DATABASE MODULE - AUTO DB CREATION + CRUD
# ============================================================

# LIBRARIES:
# pandas: Used for data manipulation and reading/writing to SQL via DataFrames.
# sqlalchemy: The SQL toolkit that provides the 'engine' to connect to the database.
# urllib.parse.quote_plus: Essential for encoding passwords that contain special characters (like '@').
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# ============================================================
# DATABASE CONFIGURATION
# ============================================================
# Dictionary containing connection parameters: credentials, host address, and target DB name.
DB_CONFIG = {
    "username": "postgres",
    "password": "@Gorgeous2703",
    "host": "localhost",
    "port": "5432",
    "database": "fedex"   # NEW DATABASE NAME
}


# ============================================================
# INTERNAL: CREATE DATABASE IF NOT EXISTS
# ============================================================
def _create_database_if_not_exists():
    """
    Connects to default 'postgres' DB and creates target DB if missing
    """
    # Encodes password to ensure special characters (e.g., '@') don't break the Connection String.
    encoded_password = quote_plus(DB_CONFIG["password"])

    # Connect to default system database
    # An 'admin' engine is needed because you cannot create a database while connected to it.
    # isolation_level="AUTOCOMMIT" is required for executing CREATE DATABASE commands.
    admin_engine = create_engine(
        f"postgresql+psycopg2://{DB_CONFIG['username']}:{encoded_password}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/postgres",
        isolation_level="AUTOCOMMIT"
    )

    with admin_engine.connect() as conn:
        # Queries the PostgreSQL system catalog (pg_database) to check if the DB name exists.
        result = conn.execute(
            text(
                "SELECT 1 FROM pg_database WHERE datname = :dbname"
            ),
            {"dbname": DB_CONFIG["database"]}
        ).scalar()

        # Logic to create the DB if it wasn't found in the system catalog.
        if not result:
            conn.execute(
                text(f'CREATE DATABASE "{DB_CONFIG["database"]}"')
            )
            print(f"✓ Database '{DB_CONFIG['database']}' created")
        else:
            print(f"✓ Database '{DB_CONFIG['database']}' already exists")


# ============================================================
# INTERNAL: CREATE ENGINE
# ============================================================
def _get_db_engine():
    """
    Returns SQLAlchemy engine for target database
    """
    # First, ensures the physical database exists before attempting to connect to it.
    _create_database_if_not_exists()

    # Re-encode password for the specific application connection.
    encoded_password = quote_plus(DB_CONFIG["password"])

    # Creates the SQLAlchemy engine using the psycopg2 driver for the target 'fedex' database.
    engine = create_engine(
        f"postgresql+psycopg2://{DB_CONFIG['username']}:{encoded_password}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )

    return engine


# ============================================================
# PUSH DATAFRAME TO POSTGRES
# ============================================================
def save_dataframe_to_postgres(
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "append"
):
    """
    Save DataFrame to PostgreSQL (auto DB creation supported)
    """

    # Initializes the connection engine.
    engine = _get_db_engine()

    try:
        # Uses pandas 'to_sql' to map the DataFrame schema to a SQL table and insert data.
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists=if_exists, # 'replace' (drop/recreate) or 'append' (add to existing).
            index=False          # Prevents pandas index from being saved as a separate column.
        )

        print(f"✓ Data saved to table: {table_name}")
        print(f"  Rows: {len(df)} | Columns: {len(df.columns)} | Mode: {if_exists}")

    except Exception as e:
        # Error handling to catch connection issues or schema mismatches.
        print(f"✗ Failed to save data to table '{table_name}'")
        raise e


# ============================================================
# FETCH DATA FROM POSTGRES
# ============================================================
def fetch_data_from_postgres(table_name: str) -> pd.DataFrame:
    """
    Fetch full table into DataFrame
    """

    # Initializes the connection engine.
    engine = _get_db_engine()

    # Define the SQL query.
    query = f"SELECT * FROM {table_name}"
    
    # Executes the query and wraps the result set into a pandas DataFrame.
    df = pd.read_sql(query, engine)

    print(f"✓ Data fetched from table: {table_name}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    return df


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    # Entry point of the script for manual testing.
    print("\n" + "=" * 60)
    print("POSTGRES AUTO DATABASE CREATION TEST")
    print("=" * 60)

    # 1. READ: Import raw data from a CSV file.
    df = pd.read_csv("data/fedex.csv")
    print(f"✓ CSV loaded: {len(df)} rows")

    # 2. WRITE: Push the CSV data into the 'fedex' database.
    save_dataframe_to_postgres(
        df=df,
        table_name="shipping_data",
        if_exists="replace"
    )

    # 3. READ-BACK: Verify the data was stored correctly by fetching it back.
    fetched_df = fetch_data_from_postgres("shipping_data")
    print("\nSample data:")
    print(fetched_df.head()) # Display the first 5 rows of the retrieved data.
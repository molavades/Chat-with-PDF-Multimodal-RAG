import snowflake.connector

# Snowflake connection parameters
SNOWFLAKE_ACCOUNT = "bvqudly-fub18765"  # Replace with your account identifier
SNOWFLAKE_USER = "viswanathraju"        # Replace with your username
SNOWFLAKE_PASSWORD = "Bigdata@8989"     # Replace with your password
SNOWFLAKE_DATABASE = "cfa"              # Replace with your database name
SNOWFLAKE_SCHEMA = "public"             # Replace with your schema name
SNOWFLAKE_WAREHOUSE = "COMPUTE_WH"      # Replace with your warehouse name

def test_snowflake_connection():
    """Test connection to Snowflake and print the status."""
    conn = None
    try:
        # Establish a connection to Snowflake
        conn = snowflake.connector.connect(
            account=SNOWFLAKE_ACCOUNT,
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA
        )
        
        # If connection is successful, print success message
        print("Connection to Snowflake was successful!")

    except Exception as e:
        print(f"Failed to connect to Snowflake. Error: {e}")

    finally:
        # Close connection if it was created
        if conn:
            conn.close()
            print("Connection closed.")

# Run the test function
test_snowflake_connection()

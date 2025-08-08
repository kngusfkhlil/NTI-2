import pyodbc
import pandas as pd

# Define connection parameters
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    "SERVER=(localdb)\MSSQLLocalDB;"
    'DATABASE=NTI;'
    'Trusted_Connection=yes;'
)
# Query your table
query = "SELECT * FROM dbo.heart_cleveland_upload"
df = pd.read_sql(query, conn)

conn.close()


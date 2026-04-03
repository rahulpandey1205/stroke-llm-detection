import sqlite3
import pandas as pd

# connect to database
conn = sqlite3.connect("stroke_ai.db")

# read patient table
df = pd.read_sql_query("SELECT * FROM patients", conn)

# print data
print("\nStored Patient Records:\n")
print(df)

conn.close()
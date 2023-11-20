"""Query the database"""

import sqlite3
from prettytable import PrettyTable

def update_region_column(cursor):
    cursor.execute("""
        UPDATE WorldSmallDB
        SET region = 
            CASE 
                WHEN region = 'C&E Europe' THEN 'Central and Eastern Europe'
                WHEN region = 'N. America' THEN 'North America'
                WHEN region = 'S. America' THEN 'South America'
                WHEN region = 'W. Europe' THEN 'Western Europe'
                ELSE region
            END;
    """)
    print("\nColumn 'region' updated successfully.")

def query():
    conn = sqlite3.connect("/workspaces/IDS706_DataEngineering_BarbaraFlores_Miniproject5/data/WorldSmallDB.db")
    cursor = conn.cursor()

    print("\nLet's quickly review our database. Let's take a sample of how it is constructed.\n")
    cursor.execute("SELECT * FROM WorldSmallDB ORDER BY RANDOM() LIMIT 5")
    print_table(cursor, cursor.fetchall())

    print("\nHow many records per continent does our database have?\n")
    cursor.execute(
        "SELECT region, COUNT(*) AS N FROM WorldSmallDB GROUP BY region"
    )
    print_table(cursor, cursor.fetchall())

    print("We are going to transform the 'region' column to make it more explanatory, replacing 'C&E Europe' with 'Central and Eastern Europe', 'N. America' with 'North America',\n'S. America' with 'South America', and 'W. Europe' with 'Western Europe.")
    update_region_column(cursor)

    print("\nHow does Gross Domestic Product per capita behave in 2008 in each continent? What are its mean, maximum, and minimum values?\n")
    cursor.execute(
        "SELECT region, AVG(gdppcap08), MIN(gdppcap08), MAX(gdppcap08) FROM WorldSmallDB GROUP BY region"
    )
    print_table(cursor, cursor.fetchall())
    conn.close()

def print_table(cursor, data):
    table = PrettyTable()
    table.field_names = [i[0] for i in cursor.description]
    for row in data:
        table.add_row(row)
    print(table)
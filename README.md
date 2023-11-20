
[![CI](https://github.com/nogibjj/IDS706_DataEngineering_BarbaraFlores_Miniproject5/actions/workflows/cicd.yml/badge.svg)](https://github.com/nogibjj/IDS706_DataEngineering_BarbaraFlores_Miniproject5/actions/workflows/cicd.yml)

IDS706_DataEngineering_BarbaraFlores_Miniproject5
## 📂 Python Script interacting with SQL Database

The goal of this miniproject is to create a Python script that effectively interacts with a SQL database.

In this miniproject, the following activities were performed:

### 1. Connect to a SQL database. 

In particular, the [world-small.csv](https://raw.githubusercontent.com/sejdemyr/sejdemyr.github.io/master/r-tutorials/basics/data/world-small.csv) database was used, which was employed in the `"Practical Data Science"` class taught by Nick Eubank. This database contains information about some countries, their regions, and their values for `Polity IV` and `gdppcap08`.

- The `polityIV` variable in this dataset is an expert score for a country's authoritarianism. Traditionally, values of -10 represent extreme autocracies, while values of 10 denote consolidated liberal democracies. However, in this dataset, they have been rescaled to range from 0 to 20, where 0 represents an extreme autocracy, and 20 represents a consolidated liberal democracy.

- The variable `gdppcap08` represents the GDP per Capita values for countries in the year 2008.

With the [extract.py](https://github.com/nogibjj/IDS706_DataEngineering_BarbaraFlores_Miniproject5/blob/main/mylib/extract.py) scrypt, we extract a CSV file from [an extrenal url](https://raw.githubusercontent.com/sejdemyr/sejdemyr.github.io/master/r-tutorials/basics/data/world-small.csv) and save its content to [a local path](https://github.com/nogibjj/IDS706_DataEngineering_BarbaraFlores_Miniproject5/blob/main/data/WorldSmall.csv). The extract function utilizes the `requests` library to make an HTTP request and then it saves the response content.

### 2. Perform CRUD operations

In this project, various CRUD operations were carried out, such as:

#### Create
- In the file [transform_load.py](https://github.com/nogibjj/IDS706_DataEngineering_BarbaraFlores_Miniproject5/blob/main/mylib/transform_load.py), the **Create** operation is performed to create the `WorldSmallDB` database along with its respective records. 

#### Read
- In the file [transform_load.py](https://github.com/nogibjj/IDS706_DataEngineering_BarbaraFlores_Miniproject5/blob/main/mylib/transform_load.py), we **Read** the specified CSV file [WorldSmall.csv](https://github.com/nogibjj/IDS706_DataEngineering_BarbaraFlores_Miniproject5/blob/main/data/WorldSmall.csv) using the function `with open(dataset, 'r', encoding='utf-8', newline='')`.

- In the file main.py: https://github.com/nogibjj/IDS706_DataEngineering_BarbaraFlores_Miniproject5/blob/main/mylib/main.py, the  **Read**  operation is performed when we read the WorldSmallDB database using the `SELECT` statement.

#### Update
- In the file main.py: https://github.com/nogibjj/IDS706_DataEngineering_BarbaraFlores_Miniproject5/blob/main/mylib/main.py, the  **Update**  operation is performed when we modify the WorldSmallDB database using the `UPDATE` statement.

#### Delete
- In the file [transform_load.py](https://github.com/nogibjj/IDS706_DataEngineering_BarbaraFlores_Miniproject5/blob/main/mylib/transform_load.py), we **DELETE** the `WorldSmallDB` database in case it does not exist previously, before creating it again.

### 3. Perform SQL queries
In the [main.py](https://github.com/nogibjj/IDS706_DataEngineering_BarbaraFlores_Miniproject5/blob/main/main.py) script, different queries are performed, which:

- SELECT a sample of the data.
- Perform a GROUP BY operation to count the number of records we have per continent.
- UPDATE the "region" field to make it more descriptive.
- Calculate the average, maximum, and minimum GDP per capita by continent in our database, using GROUP BY and aggregation functions.

### 4. Results

Finally, you can view the screenshot of successful database operations at the following prints:

![SuccessfuLDatabaseOperations](https://raw.githubusercontent.com/nogibjj/IDS706_DataEngineering_BarbaraFlores_Miniproject5/main/images/SuccessfuLDatabaseOperations.png)

### 4. ETL With SQL Diagram

![ETL With SQL Diagram](https://raw.githubusercontent.com/nogibjj/IDS706_DataEngineering_BarbaraFlores_Miniproject5/main/images/ETLWithSQLite.png)


### 5. Reference
https://github.com/nogibjj/sqlite-lab

"""
We will use a 'small world' database from Nick Eubank's Class, which is utilized for the 'Practical Data Science' class
"""
import requests

def extract(url="https://raw.githubusercontent.com/nickeubank/practicaldatascience/master/Example_Data/world-small.csv", 
            file_path="data/WorldSmall.csv"):
    """"Extract a url to a file path"""
    timeout = 100
    with requests.get(url, timeout=timeout) as r:
        with open(file_path, 'wb') as f:
            f.write(r.content)
    return file_path
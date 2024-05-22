import csv
import zlib
import gzip
import pickle

def load_q_table(filename):
    """Load a Q-table from a binary file"""
    try:
        with gzip.open(filename, mode='rb') as file:
            data = pickle.load(file)
        return data

    except FileNotFoundError:
        print("Error: Could not opend .csv file")
        return dict(dict())

def save_q_table(filename, table):
    """Save a Q-table to a binary file"""
    with gzip.open(filename, mode='wb') as file:
        pickle.dump(table, file)

def save_score(filename, score, time_taken=None):
    """Save the score to a CSV file"""
    try:
        with open(filename, mode='a',newline='') as file:
            writer = csv.writer(file)

            data = [score]
            if time_taken:
                data.append(time_taken)
            
            writer.writerow(data)
    except:
        print(f"Error: Could not open {filename} file")

def convert_state(img_state):
    """Convert the state of the game to a string"""
    data = ""
    for rows in img_state:
        for pixel in rows:
            if (str(pixel).isdigit()):
                data = data + str(pixel).strip()
    data = zlib.compress(data.encode())
    return data
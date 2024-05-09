import csv
import zlib
import gzip
import pickle

def load_csv(filename):
    """Load a Q-table from a CSV file"""
    try:
        with gzip.open(filename, mode='rb') as file:
            data = pickle.load(file)
        print(data)
        return data

    except FileNotFoundError:
        print("Error: Could not opend .csv file")
        return dict(dict())

def save_csv(filename, table):
    """Save a Q-table to a CSV file"""
    with gzip.open(filename, mode='wb') as file:
        pickle.dump(table, file)

def save_score(filename, score, time_taken):
    """Save the score to a CSV file"""
    try:
        with open(filename, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([score, time_taken])
    except:
        print(f"Error: Could not open {filename} file")

def convert_state(img_state):
    data = ""
    for rows in img_state:
        for pixel in rows:
            if (str(pixel).isdigit()):
                data = data + str(pixel).strip()
    data = zlib.compress(data.encode())
    return data

# if __name__ == "__main__":
#     test = {[[1,2,4],[1,2,4]]:{2:5,4:3}}
#     save_csv("q_table.csv", test)
#     print(load_csv("q_table.csv"))
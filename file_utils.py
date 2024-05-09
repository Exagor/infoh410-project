import csv
import zlib

def load_csv(filename):
    """Load a Q-table from a CSV file"""
    try:
        # transform csv into dictionary of dictionaries
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            # Format: {img_state: {action: value}}
            data = dict()
            for rows in reader:
                img_state = rows[0]
                dict_row = dict()

                for action,value in enumerate(rows[1:]):
                    dict_row[action] = int(value)

                data[img_state] = dict_row
        print(data)
        return data

            # return {rows[0]: {rows[1]: rows[2]} for rows in reader}
    except:
        print("Error: Could not opend .csv file")
        return dict(dict())

def save_csv(filename, table):
    """Save a Q-table to a CSV file"""
    with open(filename, mode='w',newline='') as file:
        writer = csv.writer(file)
        for img_state, dict_row in table.items():
            data = [img_state]
            for key in range(6):
                if key in dict_row: #Check if there is a key for this action
                    data.append(dict_row[key])
                else: # If not, initialize it to 0.0
                    data.append(0.0)
            # Data = state, action1, ..., action6
            writer.writerow(data)

def save_score(filename, score, time_taken):
    """Save the score to a CSV file"""
    with open(filename, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([score, time_taken])

def convert_state(img_state):
    data = ""
    for rows in img_state:
        for pixel in rows:
            if (str(pixel).isdigit()):
                data = data + str(pixel).strip()
    data = zlib.compress(data.encode())
    return data

if __name__ == "__main__":
    test = {[[1,2,4],[1,2,4]]:{2:5,4:3}}
    save_csv("q_table.csv", test)
    print(load_csv("q_table.csv"))
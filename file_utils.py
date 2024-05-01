import csv

def load_csv(filename):
    try:
        # transform csv into dictionary of dictionaries
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            return {rows[0]: {rows[1]: rows[2]} for rows in reader}
    except:
        return dict(dict())

def save_csv(filename, table):
    with open(filename, mode='w') as file:
        writer = csv.writer(file)
        for key, value in table.items():
            for key2, value2 in value.items():
                writer.writerow([key, key2, value2])
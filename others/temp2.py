import csv
a = [['m',1],['n',2],['w',3]]

import csv
filePath = 'temp.csv'

rows = [i for i in a]
with open(filePath, "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

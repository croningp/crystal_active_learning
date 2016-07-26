import csv
import numpy as np


def read_data(csv_filename):

    X = []
    y = []
    with open(csv_filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i > 0:  # first line is header
                X.append([float(x) for x in row[0:4]])
                y.append(int(row[4]))

    return np.array(X), np.array(y)


headers = ['H2O [ml]', 'HClO4 [ml]', 'NH2NH2.HCl [ml]', 'Ce(NO3)3 &Na2MoO4 [ml]', 'crystals']


def write_data(csv_filename, X, y, max_digit=3):

    with open(csv_filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(headers)
        for i in range(X.shape[0]):
            row = [round(x, max_digit) for x in X[i, :]]
            if i < y.shape[0]:
                row.append(int(y[i]))
            writer.writerow(row)

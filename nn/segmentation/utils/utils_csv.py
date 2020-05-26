#!/usr/bin/python3

import os.path
import csv

def parse_csv(csv_path):
    """parse csv file

    return: a list like
    [
    (index, image path, label path or class, [slice_index1, slice_index2, ...])
    ...
    ]
    all pathes are absolute path in the list
    no slice index indicates all slices are available

    csv_path: csv file path
    a row of a acceptable csv file looks like:
    id(usually ignored), image path, label path or class, \
            index1 index2 ...(seperated by space, \
            heading/tailing space will be ignored)

    Note: In csv file, index starts from 1.
    """
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        csv_list = []
        for row in csv_reader:
            assert len(row) == 4, 'invalid cvs format'
            subject_id = row[0]
            image = os.path.join(os.path.dirname(csv_path), row[1])
            try:
                label = int(row[2])
            except Exception as err:
                label = os.path.join(os.path.dirname(csv_path), row[2])
            # In csv file, slice index starts from 1.
            indice = list(map(lambda x: int(x)-1, row[3].strip().split()))
            if len(indice) == 0:
                indice = None
            csv_list.append([subject_id, image, label, indice])
    return csv_list

if __name__ == '__main__':
    import sys
    content = parse_csv(sys.argv[1])
    print(*list(map(str, list(content))), sep='\n')

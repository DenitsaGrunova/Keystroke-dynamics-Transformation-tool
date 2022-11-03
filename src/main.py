#!/usr/bin/env python
import os
import sys
import json
import arff
import time
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

#Διαγράφω τις τελευταίες εκδόσεις αν έχω δημιουργήσει ξανά φάκελο result
def delete_previous_versions():
    results_dir = 'results'
    if Path(results_dir).is_dir():
        print("INFO:Remove previous results folder")
        shutil.rmtree(results_dir)
    os.mkdir(results_dir)

# Φορτώνω όλα τα αρχεία txt των χρηστών και ενώνω όλα τα dataset ελέγχοντας ποια κουμπιά έχουν χρησιμοποιηθεί από τους χρήστες
def load_all_files():
    datasets = []
    classes = []
    # μπαίνω στο φάκελο και ελέγχω ποια αρχεία είναι txt
    for file in Path('data').rglob('*.txt'):
        # ανοίγω το αρχείο,το διαβάζω και βλέπω μία μία τις γραμμές του κώδικα
        with open(file, 'r', encoding="utf8") as f:
            lines = f.readlines()[:7]
            # μορφοποιώ τις πρώτες 7 γραμμές του αρχείου txt σε dictionary
            classes.append(
                {json.loads(i)[0:i.index(':')-1]: json.loads(i)[i.index(':')+1:] for i in lines})
        # φορτώνω όλα τα dataset και παραλείπω τις πρώτες 7 γραμμές
        dataset = pd.read_csv(file, skiprows=7, header=None)
        datasets.append(dataset)
        # παίρνω μια σειρά από στοιχεία και τα κάνω ένα
    return classes, pd.concat(datasets)

# πηγαίνω σε όλα τα dataset και κάνω καταμέτρηση το πόσες φορές κ έχει πατηθεί το πλήκτρο
def data_seperation(k):
    data = concat_dataset.values
    counter = defaultdict(int)
    for x in range(len(data)):
        button_dn = np.empty((3, 0))
        if data[x][3] == 'dn':
            button_dn = data[x]
            for y in range(x+1, len(data)):
                if (button_dn[0] == data[y][0]) and (button_dn[1] == data[y][1]) and (data[y][3] == 'dn'):
                    break
                elif (button_dn[0] == data[y][0]) and (button_dn[1] == data[y][1]) and (data[y][3] == 'up'):
                    counter[data[x][0]] += 1
                    break
    return [key for key, value in counter.items() if value > k]

# κάνω ζευγάρια ελέγχοντας το dn και ψάχνω το up
def find_pairs():
    pairs = []
    for x in range(len(np_selected_data)):
        # δημιουργώ πίνακα με 4 θέσεις-key,ημερομηνία,διάρκεια και state
        button_dn = np.empty((3, 0))
        button_up = np.empty((3, 0))
        if np_selected_data[x][3] == 'dn':
            button_dn = np_selected_data[x]
            # ελέγχω τα dn αν είναι το ένα κάτω απ το άλλο και αν έχουν ίδια ημερομηνια,κουμπί και state και τα παραλείπω
            for y in range(x+1, len(np_selected_data)):
                if (button_dn[0] == np_selected_data[y][0]) and (button_dn[1] == np_selected_data[y][1]) and (np_selected_data[y][3] == 'dn'):
                    break
                elif (button_dn[0] == np_selected_data[y][0]) and (button_dn[1] == np_selected_data[y][1]) and (np_selected_data[y][3] == 'up'):
                    button_up = np_selected_data[y]
                    break
        # ελέγχω αν οι μεταβλητές dn και up είναι γεμάτες τότε τα προσθέτω τα ζευγάρια σε έναν πίνακα pairs
        if button_dn.size != 0 and button_up.size != 0:
            pairs.append((button_dn, button_up))
    return pairs

# Πηγαίνω στον πίνακα pairs και βρίσκω τη διαφορά
def calculate_duration():
    results = defaultdict(list)
    for pair in pairs:
        up_btn_duration = int(pair[1][2])
        dn_btn_duration = int(pair[0][2])
        # να έχουν το ίδιο κλειδί το dictionary και να υπάρχουν σε μία λίστα όλες οι διαφορές απ τις πράξεις μεταξύ up-dn
        results[pair[0][0]].append(np.subtract(
            up_btn_duration, dn_btn_duration))
    return results

#Βρίσκω τον ΜΟ για κάθε χρήστη,για το κουμπί του-εάν δεν έχει πατήσει πλήκτρο βγαζει το ερωτηματικό
def calculate_average():
    # παίρνω το όνομα του καθενός και αφαιρω την τελεια από το όνομα του αρχείου
    file_name = file.name[0:file.name.index('.')]
    # δημιουργώ ενα dictionary με όλους τους μέσους όρους και το αρχικοποιώ με ερωτηματικά
    average = {i: '?' for i in seperated_data}
    # στην αρχή είναι όλα ερωτηματικά και μετά όταν έχω αποτέλεσμα μπαίνει στη θέση του ερωτηματικού
    for key, values in duration.items():
        average.update({int(key): round(np.average(values), 2)})
    return dict({file_name: list(average.values())})

# δημιουργία και αποθήκευση του αρχείου csv
def save2csv(final_df, rel):
    final_df.to_csv(f'results/results-{rel}.csv', mode='a', header=True)
    print(f'INFO: Saved to results-{rel}.csv')

# δημιουργία των attributes,values,κλάση του αρχείου arff
def save2arff(final_df, class_attrs, rel):
    attributes = [(str(i), 'NUMERIC')
                  for i in final_df.columns]
    attributes.pop()
    attributes.append(('class', list(set(class_attrs))))

    with open(f'results/results-{rel}.arff', "w", encoding="utf8") as f:
        arff.dump({
            'attributes': attributes,
            'data': final_df.values,
            'relation': rel,
            'description': ''
        }, f)
    print(f'INFO: Saved to results-{rel}.arff')

# ελέγχω για να κάνω την τελευταία κλάση για τα αρχεία arff
def finalize_and_save():
    min_ages = ["18-25", "26-35", "36-45"]
    min_edu = ['ISCED-2', 'ISCED-3', 'ISCED-5']

    for rel in ['Age', 'Educational Level', 'Mother Tongue']:
        class_attrs = []
        row_names = []
        updated_df = []
        updated_columns = [*df.columns, 'class']

        for class_name in classes:
            if rel == 'Age':
                class_attrs.append(
                    "45-" if class_name[rel] in min_ages else "46+")
            elif rel == 'Educational Level':
                class_attrs.append(
                    "non-academic" if class_name[rel] in min_edu else "academic")
            elif rel == 'Mother Tongue':
                class_attrs.append(class_name[rel])
      # αποθηκεύω την κλάση στο τέλος της σειράς
        for row in range(len(df.values)):
            row_names.append(df.index[row])
            updated_row = np.append(df.values[row], [class_attrs[row]])
            updated_df.append(updated_row)
        final_df = pd.DataFrame(
            updated_df, columns=updated_columns, index=row_names)

        save2csv(final_df, rel)
        save2arff(final_df, class_attrs, rel)

# καλώ μία μία τις συναρτήσεις
if __name__ == "__main__":
    SEPERATOR_VALUE = 5

    if not Path('data').is_dir():
        print("ERROR: Data folder doesn't exist.Exiting....")
        time.sleep(2)
        sys.exit()
    if len(os.listdir("data")) == 0:
        print("ERROR: Data folder is empty.Exiting....")
        time.sleep(2)
        sys.exit()
    delete_previous_versions()
    res = {}
    classes, concat_dataset = load_all_files()
    print(f'INFO: Loaded {len(classes)} classes')
    print('INFO: Seperate data')
    seperated_data = data_seperation(SEPERATOR_VALUE)
    print('INFO:Sort data')
    seperated_data.sort()
    for file in Path('data').rglob('*.txt'):
        print('\n')
        print('*'*40)
        print(f'INFO: Load file: {file.name}')
        dataset = pd.read_csv(file, skiprows=7, header=None)
        selected_data = dataset.loc[dataset[0].isin(seperated_data)]
        np_selected_data = selected_data.values
        print(f'INFO: Find pairs')
        pairs = find_pairs()
        print(f'INFO: Calculate duration')
        duration = calculate_duration()
        print(f'INFO: Calculate average')
        print('*'*40)
        average = calculate_average()
        res = res | average
    df = pd.DataFrame.from_dict(res, columns=seperated_data, orient="index")
    finalize_and_save()








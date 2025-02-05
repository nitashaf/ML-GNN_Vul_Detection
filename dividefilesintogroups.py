import os
import random
import shutil


training = 70
validating = 20
testing = 10

def dividefiles(directory):
    global training, validating, testing
    os.makedirs(directory + '/training', exist_ok=True)
    os.makedirs(directory + '/validating', exist_ok=True)
    os.makedirs(directory + '/testing', exist_ok=True)


    for cwefolder in os.listdir(directory):
        cwefolder_path = os.path.join(directory, cwefolder)

        if not os.path.isdir(cwefolder_path) or cwefolder in ['training', 'validating', 'testing']:
            continue

        filedir = [os.path.join(cwefolder_path, file) for file in os.listdir(cwefolder_path)]

        random.shuffle(filedir)

        os.makedirs(f'{directory}/training/{cwefolder}', exist_ok=True)
        os.makedirs(f'{directory}/validating/{cwefolder}', exist_ok=True)
        os.makedirs(f'{directory}/testing/{cwefolder}', exist_ok=True)

        trainingamt = int(len(filedir) * training / 100)
        validatingamt = int(len(filedir) * validating / 100)
        #testingamt = int(len(filedir) * testing / 100) # alternatively this should be whatever is left

        training_files = filedir[:trainingamt]
        validating_files = filedir[trainingamt:trainingamt + validatingamt]
        testing_files = filedir[trainingamt + validatingamt:]

        for file in training_files:
            shutil.copy(file, f'{directory}/training/{cwefolder}')
        for file in validating_files:
            shutil.copy(file, f'{directory}/validating/{cwefolder}')
        for file in testing_files:
            shutil.copy(file, f'{directory}/testing/{cwefolder}')

if __name__ == '__main__':
    Folders = ['AST', 'CFGAST', 'CFGDDGAST', 'ASTDDG']
    for folder in Folders:
        print("Processing > ", folder)
        dividefiles(f'GraphData/{folder}')
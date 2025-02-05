import os
import shutil

ALL_FOLDER_TYPES = ['AST', 'CFGAST', 'CFGDDGAST', 'ASTDDG']

for folderthing in ALL_FOLDER_TYPES:
    print("Working on folder: ", folderthing)
    current_dir = f'GraphData/ALL/{folderthing}/JulietSARD/juliet-test-suite-c/bin'

    directory_to_move_to = f'GraphData/{folderthing}'

    for folder, subfolders, files in os.walk(current_dir):
        if 'CWE' in folder:
            folder_name = folder.split('/')[-1]
            new_folder_dir = os.path.join(directory_to_move_to, folder_name)
            print(new_folder_dir)
            if not folder_name.__contains__('good') and not folder_name.__contains__('bad'):
                os.makedirs(new_folder_dir, exist_ok=True)

            for file in files:
                print(file)
                if 'good' in file:
                    # print('good >>>> ', file)
                    good_dir = os.path.join(directory_to_move_to, 'not_vulnerable')
                    os.makedirs(good_dir, exist_ok=True)
                    source_file = os.path.join(folder, file)
                    destination_file = os.path.join(good_dir, file)
                    shutil.move(source_file, destination_file)
                    print(f'moved {source_file} to {destination_file}')
                    # move to the good file
                # move all the binary files to the above f'{directory_to_move_to}/{folder}'
                if 'bad' in file:
                    # do the moving here
                    # print('bad >>>> ', file)
                    cwe = ''
                    for name in folder.split('/'):
                        if 'CWE' in name:
                            cwe = name
                    print(f'{folder} and {cwe}')
                    print(new_folder_dir)
                    if 'bad' in new_folder_dir:
                        new_folder_dir = new_folder_dir.replace('bad', cwe)
                        source_file = os.path.join(folder, file)
                        destination_file = os.path.join(new_folder_dir, file)
                        shutil.move(source_file, destination_file)
                        print(f'moved {source_file} to {destination_file}')
                    else:
                        source_file = os.path.join(folder, file)
                        destination_file = os.path.join(new_folder_dir, file)
                        shutil.move(source_file, destination_file)
                        print(f'moved {source_file} to {destination_file}')

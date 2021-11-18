import numpy as np

# read songs file and split by new rows
songs = open("songs.txt", "r")
content = songs.read()
content_list = content.splitlines()
print(songs.read())

#create dictionary
dictionary_structure = dict()
subset1 = dict()

#getting the right songs and adding them to a file
for line in content_list:
    split_by_semicolon = line.split(';')
    for semicolon in split_by_semicolon:
        dictionary_structure[semicolon] = dictionary_structure.get(semicolon, 0) + 1
for key in list(dictionary_structure):
    if dictionary_structure.get(key) > 18:
        subset1[key] = dictionary_structure[key]

# write to file
for key in dictionary_structure:
    openItemsFile = open("oneItems.txt", "a")
    double_dot = ':'
    line = str(dictionary_structure[key])+double_dot+key
    openItemsFile.writelines(line)
    print(dictionary_structure[key],':',key)

# print final result
print('Length:',len(subset1))


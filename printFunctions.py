
def printDictionaryOnFile(dictionary_name, dictionary, file):
    file.write(dictionary_name + "\n")
    for key in dictionary:
        file.write("    " + str(key) + ": " + str(dictionary[key]) + "\n")
    file.write("\n")

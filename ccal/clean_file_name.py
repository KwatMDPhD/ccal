def clean_file_name(file_name):

    file_name_clean = ""

    for character in file_name:

        if character.isalnum():

            file_name_clean += character

        else:

            file_name_clean += "_"

    return file_name_clean

from os.path import join


def create_gitkeep(directory_path):

    gitkeep_file_path = join(directory_path, ".gitkeep")

    open(gitkeep_file_path, mode="w").close()

def normalize_git_url(git_url):

    for str in ("/", ".git"):

        if git_url.endswith(str):

            git_url = git_url[: -len(str)]

    return git_url

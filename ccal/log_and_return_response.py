def log_and_return_response(response, logger=None):

    str = response.get_data().decode().strip()

    if logger is None:

        print(str)

    else:

        logger.debug(str)

    return response

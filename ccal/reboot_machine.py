from .subprocess_ import run_command


def reboot_machine():

    run_command("sudo reboot")

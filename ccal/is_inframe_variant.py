def is_inframe_variant(ref, alt):

    return not ((len(ref) - len(alt)) % 3)

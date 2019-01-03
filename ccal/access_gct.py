from warnings import warn

from pandas import read_table


def read_gct(gct_file_path, drop_description=True):

    df = read_table(gct_file_path, skiprows=2)

    column_0, column_1 = df.columns[:2]

    if column_0 == "Name":

        df.set_index("Name", inplace=True)

    else:

        raise ValueError("Column 0 != 'Name'.")

    if column_1 == "Description":

        if drop_description:

            df.drop("Description", axis=1, inplace=True)

    else:

        raise ValueError("Column 1 != 'Description'")

    return df


def write_gct(df, gct_file_path, descriptions=None):

    df = df.copy()

    if df.columns[0] != "Description":

        if descriptions is not None:

            df.insert(0, "Description", descriptions)

        else:

            df.insert(0, "Description", df.index)

    df.index.name = "Name"

    df.columns.name = None

    if not gct_file_path.endswith(".gct"):

        warn("Adding '.gct' to {} ...".format(gct_file_path))

        gct_file_path += ".gct"

    with open(gct_file_path, mode="w") as gct_file:

        gct_file.writelines("#1.2\n{}\t{}\n".format(df.shape[0], df.shape[1] - 1))

        df.to_csv(gct_file, sep="\t")

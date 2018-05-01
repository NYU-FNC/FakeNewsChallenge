import pandas as pd

from os import listdir, path

data_dir = "data/"


for file in listdir(data_dir):

    if not file.endswith(".csv"):
        continue

    fp = path.join(data_dir, file)

    df = pd.read_csv(fp)

    print(fp)
    print(len(df))

    for col in ("Headline", "articleBody"):
        if col in list(df):
            text_col = col

    out_fp = path.join(data_dir, file + ".txt")

    with open(out_fp, "w") as out:

        for text in df[text_col]:

            text_all = " ".join(text.split())
            out.write(text_all + "\n")

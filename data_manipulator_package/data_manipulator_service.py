import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option("display.max_rows", None, "display.max_columns", None)


def load_data(data_set_path: str = "../data_manipulator_package/datasets"):
    # Load all our dataset to merge them
    df_p = pd.read_csv(f"{data_set_path}/Youtube01-Psy.csv")
    df_k = pd.read_csv(f"{data_set_path}/Youtube02-KatyPerry.csv")
    df_l = pd.read_csv(f"{data_set_path}/Youtube03-LMFAO.csv")
    df_e = pd.read_csv(f"{data_set_path}/Youtube04-Eminem.csv")
    df_s = pd.read_csv(f"{data_set_path}/Youtube05-Shakira.csv")
    frame = [df_p, df_k, df_l, df_e, df_s]
    keys = ["Psy", "KatyPerry", "LMFAO", "Eminem", "Shakira"]
    df_merged_keys = pd.concat(frame, keys=keys)
    df_merged_keys.to_csv(f"{data_set_path}/YoutubeSpam.csv")
    df = df_merged_keys  # simplificare denumire

    df_x = df['CONTENT']  # datele de intare
    df_y = df['CLASS'].values  # datele de iesire

    Vectorizer = CountVectorizer()
    x_data = Vectorizer.fit_transform(df_x)  # all

    # save bag of words
    saved_file = open("../train_store_package/bag_of_words.pkl", "wb")
    pickle.dump(Vectorizer, saved_file)
    saved_file.close()

    # %%train test split
    from sklearn.model_selection import train_test_split

    return train_test_split(x_data, df_y, train_size=0.70, random_state=0)

from ingest_data import ingest

#ingest a csv from a certain path
path = "../data/t_table.csv"
df = ingest(path=path, sep=";")
df.head()

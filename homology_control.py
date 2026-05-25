from utils import load_dataframe

def create_fasta_files(train_df, test_df):
    with open("hm_cntrl/train_peptides.fasta", "w") as f:
        for idx, row in train_df.iterrows():
            f.write(f">{idx}\n{row.sequence}\n")

    with open("hm_cntrl/test_peptides.fasta", "w") as f:
        for idx, row in test_df.iterrows():
            f.write(f">{idx}\n{row.sequence}\n")

# cd-hit -i train_peptides.fasta -o train_clustered -c 0.7 -n 4
# cd-hit -i test_peptides.fasta -o test_clustered -c 0.7 -n 4 -l 3

# "train_clustered.clstr"

def create_cluster(dataframe):
    cluster_map = {}
    cluster_id = None

    with open("hm_cntrl/train_clustered.clstr") as f:
        for line in f:
            line = line.strip()

            if line.startswith(">Cluster"):
                cluster_id = int(line.split()[-1])

            else:
                seq_id = line.split(">")[1].split("...")[0]
                cluster_map[int(seq_id)] = cluster_id

    dataframe["cluster"] = dataframe.index.map(cluster_map)
    return dataframe[dataframe['cluster'].notna()]


# create_fasta_files(train_df, test_df)
import os
import tqdm
import glob
import pandas as pd
import numpy as np

from src.core.compactor import Compactor
from src.core.read_ssc import read_ssc, nb_charts


def make_dataset(folder="./data/dataset/"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    compactor = Compactor()

    datacsv = {"title": [],
               "author": [],
               "level": [],
               "tempo": [],
               "single": [],
               "beat": [],
               "length": [],
               "data": []}
    
    filenames = []
    
    tot_charts = 0
    for sscfile in tqdm.tqdm(glob.glob("./data/PIU-Simfiles-main/*/*/*.ssc")):
        try:
            for idx in range(nb_charts(sscfile)):
                tot_charts += 1
                try:
                    sfile = read_ssc(sscfile, idx)
                    if len(sfile.tempo) == 0:
                        raise Exception(f"No BPM in the file {sscfile}")
                    else:
                        filename = f"{sfile.title}-{('S' if (sfile.single) else 'D') + str(sfile.level)}"
                        filenames.append(filename)
                        count = filenames.count(filename)
                        filename = f"{filename}-{count:03d}"

                        np.save(f"{folder}{filename}.npy", compactor.read_slow(sfile.data))
                        datacsv["title"].append(sfile.title)
                        datacsv["author"].append(sfile.author)
                        datacsv["level"].append(sfile.level)
                        datacsv["tempo"].append(round(sfile.tempo[0].value))
                        datacsv["single"].append(sfile.single)
                        datacsv["beat"].append(sfile.beat)
                        datacsv["length"].append(sfile.length)
                        if filename in datacsv["data"]:
                            print(f"Warning: {filename} is already used.")
                        datacsv["data"].append(filename)
                    
                except Exception as e:
                    pass # print(f"Impossible d'ouvrir la chart: {sscfile}")
        except Exception as f:
            pass # print(f"Impossible d'ouvrir le simfile: {sscfile}")
    
    for data in datacsv:
        print(data, len(datacsv[data]), '/', tot_charts)
    
    pd.DataFrame(datacsv).to_csv(f"{folder}data.csv")
    compactor.save(f"{folder}compactor.json")
    print(f"{len(compactor.labels)} labels found.")


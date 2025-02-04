import os
from pathlib import Path
import shutil
import pandas as pd
import re


dataHouse = (
    Path(os.path.expanduser("~")) / "Library/CloudStorage/Box-Box/Brunner Lab/DATA"
)
datapath = dataHouse / "SCAN_Mayo/BJH041/stimulation_mapping/CCEPs"


runsheet = pd.read_csv(datapath / "runs.csv", dtype=str)
with open(datapath/'session_dayX_R##_freq_Anode_Cathod_numTrials_stimAmp.txt', 'w') as fp:
    fp.write('the filename for this file is how the .dat files are organized')

print(runsheet.head())
sessions = set(runsheet["session"])
for s in sessions:
    df = runsheet[runsheet["session"] == s]
    days = set(df["day"])
    try:
        for d in days:
            fd = datapath / s / d
            for f in os.listdir(fd):
                if f.find('.dat')>-1:
                    fp = fd / f
                    run = re.findall(r"R\d+", f)[0]
                    run_str = re.findall(r"\d+", run)[0]
                    if run_str[0] == "0":
                        run_num = run_str[1:]
                    else:
                        run_num = run_str
                    entry = df[df["day"] == d]
                    entry = entry[entry["run"] == run_num]
                    new_fp = (
                        "_".join(list(entry.values[0][0:2]))
                        + f"_R{run_str}_"
                        + "_".join(list(entry.values[0][3:]))
                        + ".dat"
                    )
                    os.rename(fp, datapath / s / new_fp)
                    # coding is session_day_run_freq_anode_cathod_trials_stimamp
                    print(new_fp)
        shutil.rmtree(fd)
    except Exception:
        fd = datapath / s
        for f in os.listdir(fd):
            if f.find('.dat')>-1:
                fp = fd / f
                run = re.findall(r"R\d+", f)[0]
                run_str = re.findall(r"\d+", run)[0]
                if run_str[0] == "0":
                    run_num = run_str[1:]
                else:
                    run_num = run_str
                entry = df[df["day"] == f.split('_')[1]]
                entry = entry[entry["run"] == run_num]
                new_fp = (
                    "_".join(list(entry.values[0][0:2]))
                    + f"_R{run_str}_"
                    + "_".join(list(entry.values[0][3:]))
                    + ".dat"
                )
                os.rename(fp, fd / new_fp)
                # coding is session_day_run_freq_anode_cathod_trials_stimamp
                print(new_fp)    

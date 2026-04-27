# Prepare CGM data (computes CWT scalograms from real and raw glucose CSVs provided in the [MAHP hypoglycemia prediction framework](https://github.com/Mahsan-a/MAHP).)
python Data/data_utils.py --data/cgm_csvs path as data_dir and --data/scalograms as out_dir 

The datasets supported by this framework include:

| Dataset | N patients | Ages | Monitoring duration |
|---|---|---|---|
| T1DEXI | 491 | 18–70 | 28 days |
| T1DEXIP | 227 | 12–17 | 28 days |
| CL3 | 168 | ≤14 | 6–8 months |
| CL5 | 100 | 6–13 | 16–20 weeks |
| CITY | 149 | 14–25 | 26 weeks |
| PEDAP | 98 | 2–6 | 26–32 weeks |
| AIDET1D | 82 | ≥65 | 54 weeks |

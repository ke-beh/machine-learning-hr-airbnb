# Local input data

The reviewed workflow requires these original files in this directory:

| Filename | Original rows | Purpose |
|---|---:|---|
| `hr_data.csv` | 54,808 | Employee promotion prediction |
| `listings.csv` | 7,907 | Singapore listing data; the study uses Central Region |
| `Cleaned_MRT_Stations.csv` | 171 | External station coordinates |

Alternatively, point `python -m src.evaluate --data-dir "PATH"` at a folder containing them.

CSV files are deliberately excluded from Git. Original source URLs, versions,
collection dates, currency metadata, and redistribution terms were not included
in the supplied coursework folder and have not been verified. No download URL
is invented here. This is a reproducibility dependency: a new reader cannot
rerun the experiment without obtaining the same inputs legitimately.

The original `hr_data_new.csv` and `listings_new.csv` are not used by the corrected
workflow. All learned transformations are fitted on training folds from the raw
inputs. Expected input hashes are recorded in `results/metrics.json`.

The source `listings.csv` contains host names and listing names, and the HR file
contains employee identifiers. Names are never model inputs. IDs are used only
for split integrity; aggregate results and plots contain no individual records.

The MRT snapshot date is unknown, so historical alignment with listing dates
cannot be established. Nearest-station distance is straight-line distance, not
walking distance or travel time.

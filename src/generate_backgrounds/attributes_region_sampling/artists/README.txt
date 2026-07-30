========================================================================
ARTIST SAMPLING DATASET - README
========================================================================

OVERVIEW
--------
This repository contains the artist lists used as regional cues for the
generation of synthetic conversation histories in this project. Artists
were used solely as cultural/regional signals; no ranking metrics or
analytics data are included or analyzed in this study.

DATA COLLECTION AND METHODOLOGY
-------------------------------
1. Manual collection
   The top 30 artists for each region were collected manually (no
   automated scraping or data-mining tools were used) from the Viberate
   artist ranking (https://www.viberate.com), accessed on May 29, 2026.

2. Shuffling
   The collected list was randomly shuffled to remove the original
   ranking order. The file "artists_list_shuffled.csv" contains the
   result of this step. It does not contain Viberate ranking positions
   or any performance metrics.

3. Selection
   From the shuffled list, the first 3 artists per region were manually
   selected, subject to the constraint that each of the 3 belongs to a
   different genre. Selected artists are marked TRUE in the "Selected"
   column.

4. Extraction
   The file "artists_list_shuffled.csv" was uploaded to this GitHub
   repository. A script (extract_selected.py) was then used to extract
   the 3 selected artists per region, producing:
   - selected_artists.csv  (columns: region, artist)
   - selected_artists.json (artists grouped by region)

FILES
-----
- artists_list_shuffled.csv : Shuffled top-30 artist list per region
                              (name, country, region, genre, selection flag)
- extract_selected.py       : Script that extracts the selected artists
- selected_artists.csv      : Final 3 artists per region (flat table)
- selected_artists.json     : Final 3 artists per region (grouped by region)

DATA ATTRIBUTION AND CREDITS
----------------------------
The artist selection is based on the regional artist ranking by
Viberate (Viberate, d.o.o., https://www.viberate.com), accessed on
May 29, 2026. All rights to the original ranking and its underlying data
belong to Viberate, d.o.o. Data were collected manually and are used
here exclusively for academic, non-commercial research purposes as
part of a university team project. Country, genre, and label
information originates from publicly available sources.

Data source:

  Viberate. (2026). Artist ranking by country selection [Data set]. Viberate, d.o.o.
  Retrieved May 29, 2026, from https://www.viberate.com

NOTES
-----
- Regional groupings were defined by the authors of this project from 
  the names categorization.
- The ranking is dynamic and changes over time; re-collecting the data
  at a later date will not reproduce the exact same top-30 lists. The
  access date above defines the snapshot used in this study.
========================================================================
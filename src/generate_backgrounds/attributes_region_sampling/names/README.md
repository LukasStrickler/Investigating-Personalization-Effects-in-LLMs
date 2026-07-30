README — Name Perception Analysis
================================================================================

REFERENCE
---------
This analysis is based on the dataset and validation study:

    "The Perception of Names in Experimental Studies on Ethnic Origin:
     A Cross-National Validation in Europe"
    Ghekiere, A., Martiniello, B., Capistrano, D., et al. (2025).
    Scientific Data, 12, Article 1883.
    https://doi.org/10.1038/s41597-025-06153-8

The file NameSurvey_DATARAW_2026.csv is extracted from that study and
belongs to its authors and the EqualStrength project (Horizon Europe,
project number 101094527), which produced the dataset "Perceptions of
Names in Europe". The data are used here solely for research purposes;
all credit for the data collection and validation belongs to the
original authors.

-------------------
WHAT THIS CODE DOES
-------------------
The script `names_analysis.py` processes the raw survey data
(NameSurvey_DATARAW_2026.csv) in which respondents across Europe were shown
personal names and asked:

  1. Which country/countries they believe each name comes from
     (columns V001c_<Country>), and
  2. Which gender they associate with the name (V001a).

In other words, the analysis recognizes how names are PERCEIVED by the
general population — their perceived ethnic/regional origin and perceived
gender — according to the study, rather than relying only on the names'
intended origin. This matters for experimental studies (e.g. correspondence
audits on discrimination), where a name only works as a signal if people
actually perceive it as belonging to the intended group and gender.


PROCESS (STEP BY STEP)
----------------------
1. LOAD the raw survey data.

2. INTENDED GENDER: for each name, the intended gender is inferred from
   respondents who correctly identified the gender (cong_sex == 1), taking
   the most frequent answer (Male / Female / Non-binary / Unknown).

3. TIDY FORMAT: the wide country columns (V001c_...) are reshaped into one
   row per respondent x name x perceived country. Each perceived country is
   mapped to a broader perceived region (e.g. MENA, Sub-Saharan Africa,
   Hispanic Countries, Central/Eastern Europe, United Kingdom / North
   America, France, Germany, South/Central Asia).

4. REGION-LEVEL PERCEPTION: for each name x region, the script counts the
   unique respondents who placed the name in that region and computes the
   perception percentage over the name's total evaluations. It also
   determines the dominant PERCEIVED gender in each region and keeps a
   readable breakdown of the source countries behind each region count.

5. ORIGIN-MATCH BONUS (+2): each name has an intended origin assigned by
   the study's designers. If the region perceived by respondents matches
   the intended region, the name receives a small bonus of +2 percentage
   points on its perception score (adjusted_perception_pct). This slightly
   favors names that are perceived "correctly", acting mainly as a
   tie-breaker between names with similar perception rates.

6. TOP SELECTION: within each region, names are ranked by the adjusted
   perception score. The top 3 male and top 3 female names are selected,
   subject to strict "Fit" criteria:
     - the intended gender must match the gender slot,
     - the dominant perceived gender must also match,
     - first names and surnames must be unique within the region
       (shared across the male and female lists).
   Names that outscore others but fail a gender criterion are kept in the
   table with rank = 0 and a "Not fit" label, for transparency.

7. OUTPUTS are written next to the script.


OUTPUT FILES
------------
>>> THE FILES TO BE USED ARE: top4_names_per_region.* and gender_name.csv <<<

- top4_names_per_region.csv / top4_names_per_region.json
    The main deliverable. Only region + name of the FIRST 2 "Fit" names per
    gender and region, i.e. 4 names per region (2 male + 2 female). Names
    are normalized to a single space between first/middle/last names.

- gender_name.csv
    The same selected names: column A = Name, column B = Gender.
    Ready to use directly in experimental materials.

- final_summary.csv
    Full reference table behind the selection: top 3 male + 3 female per
    region with all metrics (perception %, adjusted score with origin
    bonus, dominant perceived gender, source country counts, fit status).
    Use this file to audit or understand WHY each name was selected.


HOW TO RUN
----------
1. Place NameSurvey_DATARAW_2026.csv in the same folder as names_analysis.py.
2. Requirements: pandas, numpy.
     pip install pandas numpy
3. Run:
     python names_analysis.py
   All output files are written to the same folder as the script.

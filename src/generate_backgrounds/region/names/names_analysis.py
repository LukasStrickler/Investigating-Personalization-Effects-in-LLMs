"""
Reference:
    The Perception of Names in Experimental Studies on Ethnic Origin:
    A Cross-National Validation in Europe
    (Ghekiere et al., 2025) 
"""

import json
from pathlib import Path
 
import numpy as np
import pandas as pd
 
SCRIPT_DIR = Path(__file__).parent
INPUT_CSV = SCRIPT_DIR / "NameSurvey_DATARAW_2026.csv"
 
# ── LOAD ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV, low_memory=False)

# ── MAPPINGS ──────────────────────────────────────────────────────────────────
col_to_country = {
    'V001c_France':                'France',
    'V001c_Germany':               'Germany',
    'V001c_Spain':                 'Spain',
    'V001c_UnitedKingdom':         'United Kingdom',
    'V001c_Belgium':               'Belgium',
    'V001c_TheNetherlands':        'Netherlands',
    'V001c_Switzerland':           'Switzerland',
    'V001c_Ireland':               'Ireland',
    'V001c_Portugal':              'Portugal',
    'V001c_Italy':                 'Italy',
    'V001c_Greece':                'Greece',
    'V001c_Albania':               'Albania',
    'V001c_Slovenia':              'Slovenia',
    'V001c_Croatia':               'Croatia',
    'V001c_Hungary':               'Hungary',
    'V001c_Czechia':               'Czechia',
    'V001c_Poland':                'Poland',
    'V001c_Romania':               'Romania',
    'V001c_Ukraine':               'Ukraine',
    'V001c_Russia':                'Russia',
    'V001c_Slovakia':              'Slovakia',
    'V001c_UnitedstatesofAmerica': 'United States',
    'V001c_Canada':                'Canada',
    'V001c_Morocco':               'Morocco',
    'V001c_Egypt':                 'Egypt',
    'V001c_Algeria':               'Algeria',
    'V001c_SaudiArabia':           'Saudi Arabia',
    'V001c_Turkey':                'Turkey',
    'V001c_Iran':                  'Iran',
    'V001c_Iraq':                  'Iraq',
    'V001c_DRCongo':               'DR Congo',
    'V001c_Senegal':               'Senegal',
    'V001c_Nigeria':               'Nigeria',
    'V001c_Ethiopia':              'Ethiopia',
    'V001c_Tanzania':              'Tanzania',
    'V001c_Uganda':                'Uganda',
    'V001c_Kenya':                 'Kenya',
    'V001c_Brazil':                'Brazil',
    'V001c_Mexico':                'Mexico',
    'V001c_Colombia':              'Colombia',
    'V001c_Argentina':             'Argentina',
    'V001c_Peru':                  'Peru',
    'V001c_Venezuela':             'Venezuela',
    'V001c_Suriname':              'Suriname',
    'V001c_India':                 'India',
    'V001c_Pakistan':              'Pakistan',
    'V001c_Afghanistan':           'Afghanistan',
    'V001c_Bangladesh':            'Bangladesh',
    'V001c_Kazakhstan':            'Kazakhstan',
    'V001c_Tajikistan':            'Tajikistan',
    'V001c_SriLanka':              'Sri Lanka',
}
 
country_to_region = {
    'France':         'France',
    'Germany':        'Germany',
    'United Kingdom': 'United Kingdom / North America',
    'United States':  'United Kingdom / North America',
    'Canada':         'United Kingdom / North America',
    'Hungary':        'Central/Eastern Europe',
    'Czechia':        'Central/Eastern Europe',
    'Poland':         'Central/Eastern Europe',
    'Romania':        'Central/Eastern Europe',
    'Ukraine':        'Central/Eastern Europe',
    'Russia':         'Central/Eastern Europe',
    'Slovakia':       'Central/Eastern Europe',
    'Morocco':        'MENA',
    'Egypt':          'MENA',
    'Algeria':        'MENA',
    'Saudi Arabia':   'MENA',
    'Turkey':         'MENA',
    'Iran':           'MENA',
    'Iraq':           'MENA',
    'DR Congo':       'Sub-Saharan Africa',
    'Senegal':        'Sub-Saharan Africa',
    'Nigeria':        'Sub-Saharan Africa',
    'Ethiopia':       'Sub-Saharan Africa',
    'Tanzania':       'Sub-Saharan Africa',
    'Uganda':         'Sub-Saharan Africa',
    'Kenya':          'Sub-Saharan Africa',
    'Spain':          'Hispanic Countries',
    'Brazil':         'Hispanic Countries',
    'Mexico':         'Hispanic Countries',
    'Colombia':       'Hispanic Countries',
    'Argentina':      'Hispanic Countries',
    'Peru':           'Hispanic Countries',
    'Venezuela':      'Hispanic Countries',
    'Suriname':       'Hispanic Countries',
    'Pakistan':       'South/Central Asia',
    'India':          'South/Central Asia',
    'Afghanistan':    'South/Central Asia',
    'Bangladesh':     'South/Central Asia',
    'Kazakhstan':     'South/Central Asia',
    'Tajikistan':     'South/Central Asia',
    'Sri Lanka':      'South/Central Asia',
}
 
gender_map = {1: 'Male', 2: 'Female', 3: 'Non-binary', 4: 'Unknown'}
v_cols = list(col_to_country.keys())
 
# ── INTENDED GENDER PER NAME ──────────────────────────────────────────────────
cong_rows = df[df['cong_sex'] == 1]
name_gender = (
    cong_rows.groupby('Name')['V001a']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    .map(gender_map)
    .rename('intended_gender')
)
 
# ── RESPONDENT ID ─────────────────────────────────────────────────────────────
respondent_ids = (
    df[['input_1']].drop_duplicates()
    .reset_index(drop=True)
    .reset_index()
    .rename(columns={'index': 'respondent_id'})
)
respondent_ids['respondent_id'] += 1
 
# ── TIDY: one row per respondent × name × perceived country ───────────────────
tidy = (
    df[['input_1', 'Name', 'V001a'] + v_cols]
    .melt(
        id_vars=['input_1', 'Name', 'V001a'],
        value_vars=v_cols,
        var_name='country_col',
        value_name='picked'
    )
    .query('picked == 1')
    .drop(columns='picked')
)
 
tidy['perceived_country'] = tidy['country_col'].map(col_to_country)
tidy['perceived_region']  = tidy['perceived_country'].map(country_to_region)
tidy['perceived_gender']  = tidy['V001a'].map(gender_map)
 
tidy = tidy.merge(respondent_ids, on='input_1', how='left')
tidy = tidy.merge(name_gender,    on='Name',    how='left')
 
tidy_project = tidy[tidy['perceived_region'].notna()].copy()
 
# ── TOTAL EVALUATIONS AND INTENDED ORIGIN ─────────────────────────────────────
total_evaluations = (
    df.groupby('Name')['input_1']
    .nunique()
    .reset_index()
    .rename(columns={'input_1': 'n_total_evaluations'})
)
 
intended_origin = (
    df.groupby('Name')['country_name']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    .reset_index()
    .rename(columns={'country_name': 'intended_origin_mode'})
)
 
# ── REGION-LEVEL PICKS ────────────────────────────────────────────────────────
region_picks = (
    tidy_project
    .groupby(['Name', 'intended_gender', 'perceived_region'])['respondent_id']
    .nunique().reset_index()
    .rename(columns={'respondent_id': 'category_selected_count'})
)
 
region_picks = region_picks.merge(total_evaluations, on='Name', how='left')
region_picks['category_perception_pct'] = (
    region_picks['category_selected_count'] / region_picks['n_total_evaluations'] * 100
).round(1)
 
# ── DOMINANT PERCEIVED GENDER ─────────────────────────────────────────────────
gender_order = {'Female': 0, 'Male': 1, 'Non-binary': 2, 'Unknown': 3}
 
dominant_gender = (
    tidy_project
    .drop_duplicates(['respondent_id', 'Name', 'perceived_region', 'perceived_gender'])
    .groupby(['Name', 'perceived_region', 'perceived_gender'])['respondent_id']
    .nunique().reset_index()
    .rename(columns={'respondent_id': 'gender_count'})
)
dominant_gender['gender_rank'] = dominant_gender['perceived_gender'].map(gender_order)
dominant_gender = (
    dominant_gender
    .sort_values(['gender_count', 'gender_rank'], ascending=[False, True])
    .groupby(['Name', 'perceived_region']).first().reset_index()
    .rename(columns={
        'perceived_gender': 'dominant_perceived_gender',
        'gender_count':     'dominant_gender_count',
    })
    .drop(columns='gender_rank')
)
 
# ── SOURCE COUNTRY BREAKDOWN ──────────────────────────────────────────────────
source_country_counts = (
    tidy_project
    .groupby(['Name', 'perceived_region', 'perceived_country'])['respondent_id']
    .nunique().reset_index()
    .rename(columns={'respondent_id': 'country_count'})
)
 
source_country_str = (
    source_country_counts
    .sort_values('country_count', ascending=False)
    .groupby(['Name', 'perceived_region'])
    .apply(
        lambda x: ', '.join(
            f"{row['perceived_country']} ({row['country_count']})"
            for _, row in x.iterrows()
        ),
        include_groups=False
    )
    .reset_index()
    .rename(columns={0: 'source_country_counts'})
)
 
# ── ASSEMBLE BASE TABLE ───────────────────────────────────────────────────────
base = region_picks.copy()
base = base.merge(dominant_gender,    on=['Name', 'perceived_region'], how='left')
base = base.merge(source_country_str, on=['Name', 'perceived_region'], how='left')
base = base.merge(intended_origin,    on='Name',                       how='left')
 
base['dominant_gender_pct'] = (
    base['dominant_gender_count'] / base['category_selected_count'] * 100
).round(1)
 
# ── ORIGIN-REGION ALIGNMENT BONUS ─────────────────────────────────────────────
origin_to_region = {
    'Congo':       'Sub-Saharan Africa',
    'Nigeria':     'Sub-Saharan Africa',
    'Senegal':     'Sub-Saharan Africa',
    'Morocco':     'MENA',
    'Turkey':      'MENA',
    'Pakistan':    'South/Central Asia',
    'Ukraine':     'Central/Eastern Europe',
    'Roma':        'Central/Eastern Europe',
    'Czech':       'Central/Eastern Europe',
    'Hungary':     'Central/Eastern Europe',
    'Netherlands': 'Netherlands',
    'Belgium':     'Belgium',
    'Ireland':     'Ireland',
    'Germany':     'Germany',
    'UK':          'United Kingdom / North America',
    'CH-FR':       'France',
    'CH-DE':       'Germany',
    'Spain':       'Hispanic Countries',
}
 
base['intended_region'] = base['intended_origin_mode'].map(origin_to_region)
base['origin_match'] = (
    base['intended_region'] == base['perceived_region']
).astype(int)
 
ORIGIN_BONUS = 2
base['adjusted_perception_pct'] = (
    base['category_perception_pct'] + base['origin_match'] * ORIGIN_BONUS
).round(2)
 
base['surname'] = base['Name'].str.strip().str.split().str[-1]
 
# ── TOP 3M + 3F PER REGION WITH UNIQUE NAMES ──────────────────────────────────
def top_with_unique_surnames(group, gender, used_surnames, used_firstnames, n=3):
    selected = []
    not_fit  = []
 
    ranked = group.sort_values(
        ['adjusted_perception_pct', 'category_selected_count'],
        ascending=[False, False]
    )
 
    for _, row in ranked.iterrows():
        if len(selected) >= n:
            break
 
        if row['intended_gender'] != gender:
            row = row.copy()
            row['gender_fit'] = 'Not fit - wrong intended gender'
            not_fit.append(row)
            continue
 
        if row['dominant_perceived_gender'] != gender:
            row = row.copy()
            row['gender_fit'] = 'Not fit - perceived differently'
            not_fit.append(row)
            continue
 
        firstname = row['Name'].strip().split()[0]
 
        if row['surname'] in used_surnames:
            continue
        if firstname in used_firstnames:
            continue
 
        row = row.copy()
        row['gender_fit'] = 'Fit'
        selected.append(row)
        used_surnames.add(row['surname'])
        used_firstnames.add(firstname)
 
    result_rows = []
    for rank, row in enumerate(selected, 1):
        row['rank'] = rank
        result_rows.append(row)
    for row in not_fit:
        row['rank'] = 0
        result_rows.append(row)
 
    return pd.DataFrame(result_rows)
 
 
top_rows = []
for region, group in base.groupby('perceived_region'):
    shared_surnames   = set()
    shared_firstnames = set()
 
    for gender in ['Male', 'Female']:
        top = top_with_unique_surnames(
            group, gender,
            used_surnames=shared_surnames,
            used_firstnames=shared_firstnames,
            n=3
        )
        top['selected_gender_group'] = gender
        top_rows.append(top)
 
top_names = pd.concat(top_rows, ignore_index=True)
 
top_names = top_names.drop_duplicates(
    subset=['perceived_region', 'selected_gender_group', 'Name']
)
 
top_names = top_names[[
    'perceived_region',
    'selected_gender_group',
    'rank',
    'gender_fit',
    'Name',
    'intended_gender',
    'intended_origin_mode',
    'intended_region',
    'origin_match',
    'category_selected_count',
    'n_total_evaluations',
    'category_perception_pct',
    'adjusted_perception_pct',
    'dominant_perceived_gender',
    'dominant_gender_count',
    'dominant_gender_pct',
    'source_country_counts',
]].sort_values(
    ['perceived_region', 'selected_gender_group', 'rank'],
    ascending=[True, True, True]
).reset_index(drop=True)
 
# ── SAVE MAIN OUTPUT ──────────────────────────────────────────────────────────
top_names.to_csv(SCRIPT_DIR / 'final_summary.csv', index=False, encoding='utf-8-sig')

# ── TOP 2 FIT NAMES PER GENDER AND REGION (4 per region) ─────────────────────
top2 = (
    top_names[(top_names['gender_fit'] == 'Fit') & (top_names['rank'] <= 2)]
    .sort_values(['perceived_region', 'selected_gender_group', 'rank'])
    [['perceived_region', 'Name', 'selected_gender_group']]
    .rename(columns={'perceived_region': 'region', 'Name': 'name'})
    .reset_index(drop=True)
)
 
# Normalize names: single space between first/last/middle names
top2['name'] = top2['name'].str.strip().str.split().str.join(' ')
 
top2[['region', 'name']].to_csv(
    SCRIPT_DIR / 'top4_names_per_region.csv', index=False, encoding='utf-8-sig'
)
 
with open(SCRIPT_DIR / 'top4_names_per_region.json', 'w', encoding='utf-8') as f:
    json.dump(top2[['region', 'name']].to_dict(orient='records'), f, ensure_ascii=False, indent=2)

# ── GENDER_NAME FILE: Name + Gender of the top fit names ──────────────────────
gender_name = (
    top2[['name', 'selected_gender_group']]
    .rename(columns={'name': 'Name', 'selected_gender_group': 'Gender'})
    .reset_index(drop=True)
)
 
gender_name.to_csv(SCRIPT_DIR / 'gender_name.csv', index=False, encoding='utf-8-sig')

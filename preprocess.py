import re

import pandas as pd

GAN_data_path = 'GAN_Stroop_Data.xlsx'


def _participant_to_subject_ids(participants: pd.Series) -> pd.Series:
    """If labels are 'Player N', use N-1 as subject_id; else use category codes."""

    def parse_one(s: object) -> object:
        s = str(s).strip()
        m = re.fullmatch(r"(?i)Player\s+(\d+)", s)
        return int(m.group(1)) - 1 if m else pd.NA

    parsed = participants.map(parse_one)
    if parsed.notna().all():
        return parsed.astype(int)
    return participants.astype("category").cat.codes

# new preprocessor -> styling sheet to match conditional GAN
raw = pd.read_excel(GAN_data_path, sheet_name=0, header=None).iloc[3:]
raw[0] = raw[0].ffill() # participant names are forward-filled


portion_meta = {
    # portion up into each stroop task
    "portion 1": list(range(1, 9)), #B-I -> Stroop 1
    "portion 2": list(range(9, 17)), #J-Q -> Stroop 2
    "portion 3": list(range(17, 25)), #R-Y -> Stroop 3
    "portion 4": list(range(25, 33)), #Z-AG -> Stroop 4
}

feature_cols = ['Theta', 'Alpha', 'BetaL', 'BetaH', 'Gamma', 'Arousal', 'Valence', 'Engagement']

# build out our tidy table
frames = []
for pname, cols in portion_meta.items():
    sub = raw[[0] + cols].copy()
    sub.columns = ['Participant'] + feature_cols
    sub['portion'] = pname
    frames.append(sub)

df = pd.concat(frames, ignore_index=True)

# clean our categorial DF
df['Participant'] = df['Participant'].astype(str).str.strip()
df['portion'] = df['portion'].astype(str).str.strip()

for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=feature_cols)

# append a subject id to each participant (stable for anonymized "Player N" labels)
df["subject_id"] = _participant_to_subject_ids(df["Participant"])

#run it for final col order:
df = df[['Participant', 'subject_id', 'portion'] + feature_cols]

# finally, save to updated csv
df.to_csv('cleaned_vegs_data.csv', index=False)
print("Saved new csv to cleaned_vegs_data.csv in current root directory")
'''
df = pd.read_excel(GAN_data_path, sheet_name=0, header=None)

portion_meta = {
    "portion 1": list(range(1, 9)),     # B–I
    "portion 2": list(range(9, 17)),    # J–Q
    "portion 3": list(range(17, 25)),   # R–Y
    "portion 4": list(range(25, 33)),   # Z–AG
}

columns = ['Theta', 'Alpha', 'BetaL', 'BetaH', 'Gamma', 'Arousal', 'Valence', 'Engagement']

data = []
df = df.iloc[3:]  # skip top 3 rows (titles)

df[0] = df[0].ffill()

for portion_name, cols in portion_meta.items():
    sub = df[[0] + cols].copy()
    sub.columns = ['Participant'] + columns
    sub['portion'] = portion_name
    data.append(sub)

final = pd.concat(data)
final = final.dropna(subset=columns, how='any')

final = final[['Participant', 'portion'] + columns]

final.to_csv("cleaned_vegs_data.csv", index=False)
print(" Saved to cleaned_vegs_data.csv")
'''
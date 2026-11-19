from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.float_format", lambda value: f"{value:.2f}")

df = pd.read_csv(Path(__file__).with_name("gym_members.csv"))

print(df.shape)
print(df.describe())

under_20 = int((df["monthly_extras_spend"] < 20).sum())
at_least_20 = int((df["monthly_extras_spend"] >= 20).sum())
print(f"Members spending less than $20 a month on extras: {under_20}")
print(f"Members spending $20 or more a month on extras: {at_least_20}")

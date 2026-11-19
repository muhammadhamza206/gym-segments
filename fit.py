from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def credit_net(spend):
    if spend < 20:
        return 5
    return -10


SEGMENT_NAMES = {0: "New Starters", 1: "Loyal Veterans", 2: "Solo Grinders", 3: "Class Enthusiasts"}


def segment_offer(
    age,
    visits_per_month,
    avg_visit_minutes,
    classes_per_month,
    months_member,
    monthly_extras_spend,
):
    member = pd.DataFrame(
        [
            {
                "age": age,
                "visits_per_month": visits_per_month,
                "avg_visit_minutes": avg_visit_minutes,
                "classes_per_month": classes_per_month,
                "months_member": months_member,
                "monthly_extras_spend": monthly_extras_spend,
            }
        ]
    )[features]
    scaled = scaler.transform(member)
    point = scaled[0]
    for cluster_id, center in enumerate(v1.cluster_centers_):
        distance = float(np.linalg.norm(point - center))
        print(f"Distance to cluster {cluster_id}: {distance:.4f}")
    cluster = int(v1.predict(scaled)[0])
    gets_credit = cluster == 0
    return cluster, SEGMENT_NAMES[cluster], gets_credit


df = pd.read_csv(Path(__file__).with_name("gym_members.csv"))

train, test = train_test_split(df, test_size=0.3, random_state=42)
print(train.shape)
print(test.shape)

features = [column for column in df.columns if column != "member_id"]
X_train = train[features]
X_test = test[features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ks = range(1, 9)
train_spreads = []
test_spreads = []

for k in ks:
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(X_train_scaled)
    train_spread = model.inertia_
    test_spread = -model.score(X_test_scaled)
    train_spreads.append(train_spread)
    test_spreads.append(test_spread)
    print(f"{k} {train_spread:.1f} {test_spread:.1f}")

plt.figure()
plt.plot(list(ks), train_spreads, marker="o", label="Training spread")
plt.plot(list(ks), test_spreads, marker="o", label="Test spread")
plt.xlabel("k")
plt.ylabel("Spread")
plt.legend()
plt.tight_layout()
plt.savefig(Path(__file__).with_name("elbow.png"))

v1 = KMeans(n_clusters=4, n_init=10, random_state=42)
v1.fit(X_train_scaled)

counts = pd.Series(v1.labels_).value_counts().sort_index()
for cluster, count in counts.items():
    print(f"Cluster {cluster}: {int(count)}")

centers = pd.DataFrame(
    scaler.inverse_transform(v1.cluster_centers_),
    columns=features,
).round(1)
print(centers.to_string())

df["segment"] = v1.predict(scaler.transform(df[features]))

for segment, group in df.groupby("segment"):
    count = len(group)
    average_spend = group["monthly_extras_spend"].mean()
    under_20 = int((group["monthly_extras_spend"] < 20).sum())
    total_net = int(group["monthly_extras_spend"].map(credit_net).sum())
    print(
        f"Segment {segment}: {count} members, "
        f"average monthly extras spend {average_spend:.1f}, "
        f"{under_20} spend under $20, "
        f"total monthly net {total_net}"
    )

member_nets = df["monthly_extras_spend"].map(credit_net)
baseline_net = int(member_nets.sum())
v1_net = int(member_nets.where(df["segment"] == 0, 0).sum())
print(f"Baseline monthly net: {baseline_net}")
print(f"v1 monthly net: {v1_net}")
print(f"Difference (v1 - baseline): {v1_net - baseline_net}")

cluster, segment_name, gets_credit = segment_offer(30, 4, 42, 1, 5, 10)
print(f"Cluster: {cluster}")
print(f"Segment: {segment_name}")
print(f"Credit: {gets_credit}")

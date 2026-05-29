#!/usr/bin/env python3

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

INFILE = "rtl_medium_20_results.txt"
OUT_CSV = "rtl_medium_20_results_summary.csv"
OUT_MD = "rtl_medium_20_results_summary.md"

df = pd.read_csv(INFILE, sep=r"\s+")

y_true = df["label"].astype(int).values
y_pred = df["trigger"].astype(int).values

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred)

df.to_csv(OUT_CSV, index=False)

with open(OUT_MD, "w") as f:
    f.write("# Medium CNN RTL 20-event validation\n\n")
    f.write("| Event | Label | RTL raw signed | RTL score /128 | Trigger |\n")
    f.write("|---:|---:|---:|---:|---:|\n")

    for _, row in df.iterrows():
        f.write(
            f"| {int(row['event'])} | {int(row['label'])} | "
            f"{int(row['raw12_signed'])} | {row['score']:.6f} | {int(row['trigger'])} |\n"
        )

    f.write("\n## Metrics on first 20 RTL-simulated events\n\n")
    f.write(f"- Accuracy:  {acc:.4f}\n")
    f.write(f"- Precision: {prec:.4f}\n")
    f.write(f"- Recall:    {rec:.4f}\n")
    f.write(f"- F1:        {f1:.4f}\n")
    f.write("\nConfusion matrix, rows=true labels, columns=predicted triggers:\n\n")
    f.write("```text\n")
    f.write(str(cm))
    f.write("\n```\n")

print(df)
print()
print("Metrics on 20 RTL events")
print("========================")
print("accuracy :", acc)
print("precision:", prec)
print("recall   :", rec)
print("f1       :", f1)
print("confusion matrix:")
print(cm)
print()
print(f"[+] Wrote {OUT_CSV}")
print(f"[+] Wrote {OUT_MD}")

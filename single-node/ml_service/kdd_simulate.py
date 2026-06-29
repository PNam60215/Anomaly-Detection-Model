import json, os, random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from ml_service import (
    preprocess_batch_if, preprocess_batch_ae, predict_batch,
    AE_THRESHOLD
)

# ── Cấu hình ─────────────────────────────────────────────────────────────────
TEST_FILE  = '/app/KDDTest+.txt'
ALERT_FILE = '/alerts/ml_alerts.json'
BATCH_SIZE = 512

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

# ── Đọc dữ liệu ──────────────────────────────────────────────────────────────
print("=" * 60)

df = pd.read_csv(TEST_FILE, header=None, names=COLUMNS)
df["attack"]    = (df["label"] != "normal").astype(int)
random.seed(42)
df["src_ip"]    = [f"10.0.0.{random.randint(1, 254)}" for _ in range(len(df))]
df["timestamp"] = [datetime.now().isoformat() for _ in range(len(df))]

print(f"Tổng    : {len(df)} records")
print(f"Normal  : {(df['attack']==0).sum()}")
print(f"Attack  : {(df['attack']==1).sum()}")

# ── Tiền xử lý và dự đoán batch ──────────────────────────────────────────────
print("\nĐang tiền xử lý...")
X_if = preprocess_batch_if(df)
X_ae = preprocess_batch_ae(df)
print(f"  IF features : {X_if.shape}")
print(f"  AE features : {X_ae.shape}")

print("\nĐang dự đoán...")
if_preds, ae_preds, ens_preds, if_scores, ae_re = predict_batch(
    X_if, X_ae, batch_size=BATCH_SIZE
)
y_true = df["attack"].values
print("Dự đoán xong.")

# ── In metrics ────────────────────────────────────────────────────────────────
def print_metrics(name, y_true, y_pred, scores=None):
    print(f"\n{'='*60}")
    print(f"MÔ HÌNH: {name}")
    print(f"{'='*60}")
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    if scores is not None:
        try:
            print(f"ROC-AUC   : {roc_auc_score(y_true, scores):.4f}")
        except Exception:
            pass
    print(f"\n{classification_report(y_true, y_pred, target_names=['Normal', 'Attack'])}")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

print_metrics("Isolation Forest", y_true, if_preds.astype(int),  if_scores)
print_metrics("Autoencoder",      y_true, ae_preds.astype(int),  ae_re)
print_metrics("Ensemble (AND)",   y_true, ens_preds.astype(int))

# ── Ghi alert lên Wazuh ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Ghi alert vào Wazuh...")


os.makedirs('/alerts', exist_ok=True)
alert_count = {"Isolation Forest": 0, "Autoencoder": 0, "Ensemble": 0}

with open(ALERT_FILE, 'a') as f:
    for i in range(len(df)):
        if_attack = bool(if_preds[i])
        ae_attack = bool(ae_preds[i])

        if not (if_attack or ae_attack):
            continue

        if if_attack and ae_attack:
            model_name, level = "Ensemble", 15
        elif if_attack:
            model_name, level = "Isolation Forest", 12
        else:
            model_name, level = "Autoencoder", 12

        alert = {
            "timestamp":    df.at[df.index[i], "timestamp"],
            "src_ip":       df.at[df.index[i], "src_ip"],
            "protocol":     df.at[df.index[i], "protocol_type"],
            "service":      df.at[df.index[i], "service"],
            "model":        model_name,
            "if_attack":    if_attack,
            "ae_attack":    ae_attack,
            "if_score":     round(float(if_scores[i]), 6),
            "ae_re":        round(float(ae_re[i]), 6),
            "ae_threshold": round(AE_THRESHOLD, 6),
            "label_true":   df.at[df.index[i], "label"],
            "rule": {
                "level":       level,
                "description": f"Anomaly detected by {model_name}"
            }
        }
        f.write(json.dumps(alert) + "\n")
        f.flush()
        alert_count[model_name] += 1

total = sum(alert_count.values())
print(f"\nTổng alert đã ghi : {total}")
for model, cnt in alert_count.items():
    print(f"  {model}: {cnt}")
print(f"\nKiểm tra Wazuh Dashboard để xem alert.")

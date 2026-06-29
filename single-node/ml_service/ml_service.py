import json, joblib, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore", category=UserWarning)

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")

iso_forest    = joblib.load('/app/models/isolation_forest.pkl')
freq_encoding = joblib.load('/app/models/freq_encoding.pkl')
scaler_if     = joblib.load('/app/models/scaler_if.pkl')

with open('/app/models/if_threshold.json') as f:
    IF_THRESHOLD = json.load(f)['if_threshold']

class Autoencoder(keras.Model):
    def __init__(self, input_dim=146, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.encoder = keras.Sequential([
            keras.layers.Dense(64, activation='tanh'),
            keras.layers.Dense(32, activation='tanh'),
            keras.layers.Dense(16, activation='tanh'),
            keras.layers.Dense(8,  activation='tanh'),
        ])
        self.decoder = keras.Sequential([
            keras.layers.Dense(16,        activation='tanh'),
            keras.layers.Dense(32,        activation='tanh'),
            keras.layers.Dense(64,        activation='tanh'),
            keras.layers.Dense(input_dim, activation='sigmoid'),
        ])

    def call(self, x):
        return self.decoder(self.encoder(x))

    def get_config(self):
        return {'input_dim': self.input_dim}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

autoencoder = tf.keras.models.load_model(
    '/app/models/autoencoder.keras',
    custom_objects={'Autoencoder': Autoencoder}
)
scaler_ae = joblib.load('/app/models/scaler_ae.pkl')
ohe       = joblib.load('/app/models/onehot_encoder.pkl')

with open('/app/models/model_meta.json') as f:
    AE_THRESHOLD = json.load(f)['threshold']

print(f"Models loaded.")
print(f"  IF_THRESHOLD : {IF_THRESHOLD:.6f}")
print(f"  AE_THRESHOLD : {AE_THRESHOLD:.6f}")

# ── Định nghĩa cột ────────────────────────────────────────────────────────────
NUMERIC_COLS_IF = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'dst_host_count',
    'dst_host_srv_count', 'difficulty'
]

RATE_COLS_IF = [
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

CAT_COLS = ["protocol_type", "service", "flag", "label"]

COLUMNS_AE = [
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
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "difficulty"
]

# ── Tiền xử lý batch ─────────────────────────────────────────────────────────
def preprocess_batch_if(df):
    """Tiền xử lý toàn bộ DataFrame cho Isolation Forest, trả về numpy array."""
    num_scaled   = scaler_if.transform(df[NUMERIC_COLS_IF])
    service_freq = df["service"].map(freq_encoding).fillna(0.0).values.reshape(-1, 1)
    rate_values  = df[RATE_COLS_IF].values
    flags        = df["flag"].values
    protocols    = df["protocol_type"].values
    ohe_matrix   = np.column_stack([
        (flags == "REJ").astype(int),    (flags == "RSTO").astype(int),
        (flags == "RSTOS0").astype(int), (flags == "RSTR").astype(int),
        (flags == "S0").astype(int),     (flags == "S1").astype(int),
        (flags == "S2").astype(int),     (flags == "S3").astype(int),
        (flags == "SF").astype(int),     (flags == "SH").astype(int),
        (protocols == "tcp").astype(int),(protocols == "udp").astype(int),
    ])
    return np.concatenate(
        [num_scaled[:, 0:1], service_freq, num_scaled[:, 1:], rate_values, ohe_matrix],
        axis=1
    )

def preprocess_batch_ae(df):
    """Tiền xử lý toàn bộ DataFrame cho Autoencoder, trả về numpy array."""
    ae_df          = df[COLUMNS_AE].copy()
    ae_df["label"] = df["label"].values
    numerical      = ae_df.select_dtypes(exclude='object').values
    encoded_cat    = ohe.transform(ae_df[CAT_COLS].values).toarray()
    return scaler_ae.transform(np.concatenate([numerical, encoded_cat], axis=1))

# ── Dự đoán batch ─────────────────────────────────────────────────────────────
def predict_batch(X_if, X_ae, batch_size=512):
    """
    Chạy cả 2 model trên toàn bộ dataset cùng lúc.
    Trả về: if_preds, ae_preds, ens_preds, if_scores, ae_re
    """
    # Isolation Forest
    if_scores_raw = iso_forest.score_samples(X_if)
    if_preds      = (if_scores_raw < IF_THRESHOLD)
    if_scores     = np.abs(if_scores_raw)

    # Autoencoder
    recon    = autoencoder.predict(X_ae, batch_size=batch_size, verbose=1)
    ae_re    = np.mean(np.power(X_ae - recon, 2), axis=1)
    ae_preds = (ae_re > AE_THRESHOLD)

    # Ensemble AND
    ens_preds = if_preds & ae_preds

    return if_preds, ae_preds, ens_preds, if_scores, ae_re

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

# 1. Load dataset
train_df = pd.read_csv('data/raw/UNSW_NB15_training-set.csv')
test_df = pd.read_csv('data/raw/UNSW_NB15_testing-set.csv')

# 2. Drop unused columns
drop_cols = ['id']
train_df = train_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)

# 3. Select features
selected_features = [
    'sttl', 'ct_state_ttl', 'dload', 'ct_dst_sport_ltm',
    'dmean', 'rate', 'ct_src_dport_ltm',
]

# 4. Split features & label
X_trainval = train_df[selected_features]
y_trainval = train_df['label']

X_test = test_df[selected_features]
y_test = test_df['label']

# 5. Normalize with MinMaxScaler (fit only on trainval, apply to all)
scaler = MinMaxScaler()
X_trainval_scaled = scaler.fit_transform(X_trainval)
X_test_scaled = scaler.transform(X_test)

# 6. Split train–val from training data
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval_scaled, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42
)

# 7. Save output
os.makedirs('data/split', exist_ok=True)

pd.DataFrame(X_train, columns=selected_features).assign(label=y_train.values).to_csv('data/split/train.csv', index=False)
pd.DataFrame(X_val, columns=selected_features).assign(label=y_val.values).to_csv('data/split/val.csv', index=False)
pd.DataFrame(X_test_scaled, columns=selected_features).assign(label=y_test.values).to_csv('data/split/test.csv', index=False)

print("✅ Preprocessing complete! Train, Val, Test saved in 'data/split/'")
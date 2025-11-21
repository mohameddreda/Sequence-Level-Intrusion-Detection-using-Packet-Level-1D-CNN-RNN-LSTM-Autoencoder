# src/preprocessor.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from config import Config

class DataPreprocessor:
    def __init__(self):
        self.config = Config()
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def balance_data(self, df):
        """Balance the data like in the notebook"""
        print("🔄 Balancing data...")
        
        counts_before = df[self.config.ATTACK_CAT_COLUMN].value_counts()
        print("📊 Before balancing:")
        print(counts_before)
        
        max_samples = 30000
        balanced_frames = []
        
        for attack, count in df[self.config.ATTACK_CAT_COLUMN].value_counts().items():
            subset = df[df[self.config.ATTACK_CAT_COLUMN] == attack]
            if count < max_samples:
                subset_balanced = self.resample_data(subset, n_samples=max_samples)
            else:
                subset_balanced = subset.copy()
            balanced_frames.append(subset_balanced)
        
        df_balanced = pd.concat(balanced_frames)
        counts_after = df_balanced[self.config.ATTACK_CAT_COLUMN].value_counts()
        
        print("📊 After balancing:")
        print(counts_after)
        
        return df_balanced
    
    def resample_data(self, data, n_samples):
        """Resample data for balancing"""
        return data.sample(n=n_samples, replace=True, random_state=self.config.RANDOM_STATE)
    
    def preprocess_features(self, df):
        """Preprocess features like the notebook"""
        print("🔄 Preprocessing features...")
        df_processed = df.copy()
        
        # نستخدم نفس الـ features المختارة
        selected_features = [f for f in self.config.SELECTED_FEATURES if f in df_processed.columns]
        
        # Handle missing values
        df_processed[selected_features] = df_processed[selected_features].fillna(0)
        
        # Encode categorical features إذا وجدت
        categorical_cols = df_processed[selected_features].select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le
        
        # Standardize numerical features
        numerical_cols = df_processed[selected_features].select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) > 0:
            df_processed[numerical_cols] = self.scaler.fit_transform(df_processed[numerical_cols])
        
        return df_processed[selected_features]
    
    def create_sequences(self, data, labels, df=None):
        """
        Create sequences for the autoencoder, grouped by session (5-tuple) if possible.
        If df is provided, group by session keys: src_ip, dst_ip, src_port, dst_port, proto.
        Otherwise, fallback to sliding window.
        """
        print("🔄 Creating sequences (session-based if possible)...")
        sequences = []
        sequence_labels = []

        if df is not None:
            # Try to group by session keys
            session_keys = ['srcip', 'dstip', 'sport', 'dsport', 'proto']
            available_keys = [k for k in session_keys if k in df.columns]
            if len(available_keys) == 5:
                grouped = df.groupby(available_keys)
                for _, group in grouped:
                    group_data = data[group.index]
                    group_labels = labels[group.index]
                    n_sequences = len(group_data) - self.config.SEQUENCE_LENGTH + 1
                    for i in range(n_sequences):
                        sequence = group_data[i:i + self.config.SEQUENCE_LENGTH]
                        sequence_label = group_labels[i + self.config.SEQUENCE_LENGTH - 1]
                        sequences.append(sequence)
                        sequence_labels.append(sequence_label)
                print(f"Created {len(sequences)} session-based sequences.")
                return np.array(sequences), np.array(sequence_labels)
            else:
                print("Session keys not found, using sliding window.")

        # Fallback: sliding window
        if isinstance(data, pd.DataFrame):
            data = data.values
        n_sequences = len(data) - self.config.SEQUENCE_LENGTH + 1
        for i in range(n_sequences):
            sequence = data[i:i + self.config.SEQUENCE_LENGTH]
            sequence_label = labels[i + self.config.SEQUENCE_LENGTH - 1]
            sequences.append(sequence)
            sequence_labels.append(sequence_label)
        print(f"Created {len(sequences)} sliding-window sequences.")
        return np.array(sequences), np.array(sequence_labels)
    
    def prepare_data(self, df):
        """Complete data preparation pipeline (session-based sequences if possible)"""
        # نعمل balance للبيانات أولاً
        df_balanced = self.balance_data(df)

        # نجهز الـ features
        X = self.preprocess_features(df_balanced)
        y = df_balanced[self.config.LABEL_COLUMN].values

        print(f"📊 Features: {X.shape}, Labels: {y.shape}")

        # ننشئ sequences (session-based if possible)
        X_seq, y_seq = self.create_sequences(X, y, df=df_balanced)

        # نقسم البيانات: أولاً train / test ثم قسم الـ train إلى train / val
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_seq, y_seq,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y_seq
        )

        # الآن نقسم الـ train_full إلى train و val (نحتاج val لمعايرة threshold)
        if hasattr(self.config, 'VAL_SIZE') and self.config.VAL_SIZE > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full,
                test_size=self.config.VAL_SIZE,
                random_state=self.config.RANDOM_STATE,
                stratify=y_train_full
            )
        else:
            X_train, y_train = X_train_full, y_train_full
            X_val, y_val = None, None

        # نأخذ الـ normal sequences فقط للتدريب (مبدأ الـ autoencoder)
        normal_indices = np.where(y_train == 0)[0]
        X_train_normal = X_train[normal_indices]

        print(f"🎯 Training sequences (normal only): {X_train_normal.shape}")
        print(f"🧪 Validation sequences: {None if X_val is None else X_val.shape}")
        print(f"🧪 Test sequences: {X_test.shape}")

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'X_train_normal': X_train_normal,
            'sequence_shape': X_seq.shape
        }
# src/data_loader.py
import pandas as pd
import numpy as np
import os
from config import Config

class DataLoader:
    def __init__(self):
        self.config = Config()
    
    def load_data(self):
        """Load data from all available sources"""
        try:
            # Option 1: استخدام training-set و testing-set إذا موجودين
            if self.config.USE_SPLIT_FILES and os.path.exists(self.config.TRAIN_PATH):
                print("📊 Loading split datasets...")
                train_df = pd.read_csv(self.config.TRAIN_PATH)
                test_df = pd.read_csv(self.config.TEST_PATH)
                df = pd.concat([train_df, test_df], ignore_index=True)
                print(f"✅ Training set: {train_df.shape}")
                print(f"✅ Testing set: {test_df.shape}")
            
            # Option 2: استخدام الملفات الأربعة
            else:
                print("📊 Loading from 4 CSV files...")
                dfs = []
                total_samples = 0

                # Try to read header names from the TRAIN_PATH if available
                header_names = None
                try:
                    if os.path.exists(self.config.TRAIN_PATH):
                        header_names = pd.read_csv(self.config.TRAIN_PATH, nrows=0).columns.tolist()
                except Exception:
                    header_names = None

                for path in self.config.DATA_PATHS:
                    if os.path.exists(path):
                        print(f"Loading {path}...")
                        # Many UNSW NB15 split files don't include headers — read without header
                        # and apply header names from the training-set if available.
                        if header_names:
                            df_part = pd.read_csv(path, nrows=20000, header=None, low_memory=False)
                            # If number of columns differs, only assign what's available
                            if df_part.shape[1] == len(header_names):
                                df_part.columns = header_names
                            else:
                                # If columns differ, try to set as many names as possible
                                n = min(df_part.shape[1], len(header_names))
                                df_part.columns = header_names[:n] + [f"col_{i}" for i in range(n, df_part.shape[1])]
                        else:
                            df_part = pd.read_csv(path, nrows=20000, header=None, low_memory=False)

                        dfs.append(df_part)
                        total_samples += len(df_part)
                        print(f"✅ Loaded {len(df_part)} samples from {path}")
                    else:
                        print(f"⚠️ File not found: {path}")

                if not dfs:
                    print("❌ No data files found!")
                    return None

                df = pd.concat(dfs, ignore_index=True)
                print(f"🎉 Combined dataset: {df.shape} (from {len(dfs)} files)")
            
            # نتأكد من وجود الأعمدة المطلوبة
            self._validate_columns(df)
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _validate_columns(self, df):
        """Validate that required columns exist"""
        print("🔍 Validating columns...")
        
        # نتأكد من وجود الأعمدة الأساسية
        required_columns = [self.config.LABEL_COLUMN, self.config.ATTACK_CAT_COLUMN]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️ Missing columns: {missing_columns}")
            print(f"📋 Available columns: {list(df.columns)}")
        
        # نتأكد من وجود الـ features المختارة
        available_features = [f for f in self.config.SELECTED_FEATURES if f in df.columns]
        missing_features = [f for f in self.config.SELECTED_FEATURES if f not in df.columns]
        
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
        
        print(f"🎯 Using {len(available_features)} available features out of {len(self.config.SELECTED_FEATURES)}")
    
    def get_basic_info(self, df):
        """Get basic dataset information"""
        if df is None:
            return {}
            
        info = {
            'shape': df.shape,
            'label_distribution': df[self.config.LABEL_COLUMN].value_counts() if self.config.LABEL_COLUMN in df.columns else "N/A",
            'attack_distribution': df[self.config.ATTACK_CAT_COLUMN].value_counts() if self.config.ATTACK_CAT_COLUMN in df.columns else "N/A"
        }
        
        return info
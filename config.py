# config.py
class Config:
    # Data paths - نستخدم كل الملفات الأربعة
    DATA_PATHS = [
        'data/raw/UNSW-NB15_1.csv',
        'data/raw/UNSW-NB15_2.csv',
        'data/raw/UNSW-NB15_3.csv', 
        'data/raw/UNSW-NB15_4.csv'
    ]
    
    # أو نستخدم training-set و testing-set إذا موجودين
    # The provided training/testing files include proper headers — prefer them when available.
    USE_SPLIT_FILES = True  # use the TRAIN_PATH and TEST_PATH when present
    TRAIN_PATH = 'data/raw/UNSW_NB15_training-set.csv'
    TEST_PATH = 'data/raw/UNSW_NB15_testing-set.csv'
    
    # Feature selection like the notebook
    SELECTED_FEATURES = [
        'sttl', 'swin', 'ct_dst_sport_ltm', 'id', 'dwin',
        'ct_src_dport_ltm', 'rate', 'ct_state_ttl', 'ct_srv_dst',
        'ct_srv_src', 'dtcpb', 'stcpb', 'dload', 'ct_dst_src_ltm', 'ct_src_ltm'
    ]
    
    # Model parameters
    SEQUENCE_LENGTH = 10
    LATENT_DIM = 32
    BATCH_SIZE = 256
    EPOCHS = 25
    LEARNING_RATE = 0.001
    
    # Training parameters
    TEST_SIZE = 0.2
    # Fraction of the training split to use as a validation set for threshold tuning
    VAL_SIZE = 0.1
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42
    
    # Anomaly detection
    THRESHOLD_PERCENTILE = 95
    
    # Label column
    LABEL_COLUMN = 'label'
    ATTACK_CAT_COLUMN = 'attack_cat'
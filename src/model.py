import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, 
    RepeatVector, TimeDistributed
)
from tensorflow.keras.optimizers import Adam
from config import Config

class HybridAutoencoder:
    def __init__(self):
        self.config = Config()
        self.autoencoder = None
        self.encoder = None
    
    def build_model(self, sequence_length, n_features):
        """Build the hybrid CNN+LSTM autoencoder"""
        # Encoder
        inputs = Input(shape=(sequence_length, n_features))
        
        # 1D CNN layers
        x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(32, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(2, padding='same')(x)
        
        # LSTM layers
        x = LSTM(64, activation='relu', return_sequences=True)(x)
        x = LSTM(32, activation='relu', return_sequences=False)(x)
        
        # Bottleneck
        encoded = Dense(self.config.LATENT_DIM, activation='relu')(x)
        
        # Decoder
        x = Dense(32, activation='relu')(encoded)
        x = RepeatVector(sequence_length)(x)
        
        # LSTM layers
        x = LSTM(32, activation='relu', return_sequences=True)(x)
        x = LSTM(64, activation='relu', return_sequences=True)(x)
        
        # CNN layers
        x = Conv1D(32, 3, activation='relu', padding='same')(x)
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        
        # Output
        decoded = Conv1D(n_features, 3, activation='linear', padding='same')(x)
        
        # Models
        autoencoder = Model(inputs, decoded)
        encoder = Model(inputs, encoded)
        
        autoencoder.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        self.autoencoder = autoencoder
        self.encoder = encoder
        
        return autoencoder, encoder
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.autoencoder:
            self.autoencoder.summary()
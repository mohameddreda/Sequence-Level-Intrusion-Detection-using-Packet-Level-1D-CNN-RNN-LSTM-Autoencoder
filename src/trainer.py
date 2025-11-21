import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from config import Config

class ModelTrainer:
    def __init__(self):
        self.config = Config()
        self.history = None
    
    def train(self, model, X_train_normal):
        """Train the autoencoder model"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=3, 
                min_lr=1e-6
            )
        ]
        # Save best model during training
        checkpoint_path = 'models/saved_models/autoencoder_best.h5'
        callbacks.append(
            ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=False)
        )
        
        self.history = model.fit(
            X_train_normal, X_train_normal,
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            validation_split=self.config.VALIDATION_SPLIT,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # MAE plot
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        return fig
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import (AUGMENTATION, EPOCHS, BATCH_SIZE, USE_EARLY_STOPPING,
                    EARLY_STOPPING_PATIENCE, USE_MODEL_CHECKPOINT, CHECKPOINT_PATH,
                    USE_TENSORBOARD, LOGS_DIR)
from src.utils import plot_history
import logging

logger = logging.getLogger(__name__)

def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model with data augmentation and callbacks."""
    
    # Data augmentation
    datagen = ImageDataGenerator(**AUGMENTATION)
    datagen.fit(X_train)
    
    # Callbacks
    callbacks = []
    if USE_EARLY_STOPPING:
        early_stop = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
        callbacks.append(early_stop)
    if USE_MODEL_CHECKPOINT:
        checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor='val_accuracy', save_best_only=True, mode='max')
        callbacks.append(checkpoint)
    if USE_TENSORBOARD:
        tensorboard = TensorBoard(log_dir=LOGS_DIR, histogram_freq=1)
        callbacks.append(tensorboard)
    
    # Train
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_history(history)
    
    return history

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from datetime import datetime
from tensorflow.keras.utils import to_categorical
from PIL.ExifTags import TAGS
import logging

def get_date_from_exif(image_path):
    """
    Extract the date from image EXIF data.
    Returns None if no date is found.
    """
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
            
            if not exif:
                return None
                
            # Look for various date fields in EXIF
            date_fields = [36867, 36868, 306]  # DateTimeOriginal, DateTimeDigitized, DateTime
            
            for field in date_fields:
                if field in exif:
                    date_str = exif[field]
                    try:
                        # EXIF date format: 'YYYY:MM:DD HH:MM:SS'
                        date_obj = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                        return date_obj
                    except ValueError:
                        continue
                        
            return None
            
    except Exception as e:
        logging.warning(f"Could not read EXIF from {image_path}: {str(e)}")
        return None

def create_dataset(photo_dir):
    """
    Create a dataset from photos in a directory.
    Returns images and their months (1-12) as categorical labels.
    """
    images = []
    months = []
    skipped = 0
    
    for filename in os.listdir(photo_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(photo_dir, filename)
            
            # Try to get date from EXIF
            date = get_date_from_exif(img_path)
            
            # If no EXIF date, fall back to file modification time
            if date is None:
                timestamp = os.path.getmtime(img_path)
                date = datetime.fromtimestamp(timestamp)
                logging.info(f"No EXIF date found for {filename}, using file modification time")
            
            # Load and preprocess image
            try:
                img = Image.open(img_path)
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0
                
                # Only add image if it has 3 color channels (RGB)
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    images.append(img_array)
                    months.append(date.month - 1)  # Convert to 0-11 range
                else:
                    skipped += 1
                    logging.warning(f"Skipping {filename}: not RGB format")
                    
            except Exception as e:
                skipped += 1
                logging.warning(f"Error processing {filename}: {str(e)}")
                continue
    
    if skipped > 0:
        logging.info(f"Skipped {skipped} images due to format issues or errors")
        
    if not images:
        raise ValueError("No valid images found in directory")
    
    # Convert months to one-hot encoded format
    months_categorical = to_categorical(months, num_classes=12)
    
    return np.array(images), months_categorical

def build_model():
    """
    Create a CNN model for month prediction
    """
    model = models.Sequential([
        # Base CNN architecture
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(12, activation='softmax')  # 12 outputs for months
    ])
    
    return model

def train_month_predictor(photo_dir, epochs=10):
    """
    Train a model to predict the month a photo was taken
    """
    # Prepare dataset
    images, months = create_dataset(photo_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        images, months, test_size=0.2, random_state=42
    )
    
    # Build and compile model
    model = build_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_test, y_test),
        batch_size=32
    )
    
    return model, history

def predict_month(model, image_path):
    """
    Predict the month for a single photo
    """
    # Load and preprocess image
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    month_probabilities = model.predict(img_array)[0]
    predicted_month = np.argmax(month_probabilities) + 1  # Convert back to 1-12 range
    
    # Get confidence score
    confidence = month_probabilities[predicted_month - 1]
    
    # Convert month number to name
    month_names = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    month_name = month_names[predicted_month - 1]
    
    return month_name, predicted_month, confidence

def analyze_predictions(model, test_images, true_months):
    """
    Analyze model performance with confusion matrix and per-month accuracy
    """
    predictions = model.predict(test_images)
    predicted_months = np.argmax(predictions, axis=1)
    true_months = np.argmax(true_months, axis=1)
    
    from sklearn.metrics import confusion_matrix, classification_report
    conf_matrix = confusion_matrix(true_months, predicted_months)
    class_report = classification_report(
        true_months, 
        predicted_months,
        target_names=[
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        ]
    )
    
    return conf_matrix, class_report

if __name__ == '__main__':
    model, history = train_month_predictor('photos')
    month_name, predicted_month, confidence = predict_month(model, 'photos/IMG_5175.JPG')
    print(predicted_month)

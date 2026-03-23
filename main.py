import argparse
import logging
from config import *
from src.data_loader import DataLoader
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_image
from src.utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Malaria Diagnosis CNN')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'predict'],
                        help='Mode: train, evaluate, or predict')
    parser.add_argument('--model_path', type=str, default=CHECKPOINT_PATH,
                        help='Path to model file (for evaluate/predict)')
    parser.add_argument('--image_path', type=str, help='Path to image for prediction')
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    if args.mode == 'train':
        logger.info("Starting training...")
        data_loader = DataLoader(DATA_DIR, IMG_SIZE, test_split=TEST_SPLIT, random_seed=RANDOM_SEED)
        X_train, X_test, y_train, y_test = data_loader.load_data()
        model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3))
        history = train_model(model, X_train, y_train, X_test, y_test)
        logger.info("Training completed.")

    elif args.mode == 'evaluate':
        logger.info("Evaluating model...")
        data_loader = DataLoader(DATA_DIR, IMG_SIZE, test_split=TEST_SPLIT, random_seed=RANDOM_SEED)
        _, X_test, _, y_test = data_loader.load_data()
        evaluate_model(args.model_path, X_test, y_test)
        logger.info("Evaluation completed.")

    elif args.mode == 'predict':
        if not args.image_path:
            logger.error("Please provide --image_path for prediction mode.")
            return
        logger.info(f"Predicting on image: {args.image_path}")
        result = predict_image(args.model_path, args.image_path, IMG_SIZE)
        print(f"Prediction: {result}")
        logger.info("Prediction completed.")

if __name__ == '__main__':
    main()

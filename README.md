# FruitScan AI

AI-powered fruit and vegetable image classification using a CNN trained on Fruits-360.

## Team workflow
This repository is intentionally delivered in three implementation stages:
1. Participant 1: dataset research and CNN training.
2. Participant 2: inference engine and tests.
3. Participant 3: Streamlit application and final documentation.

Each participant should add only their assigned files in their own GitHub account and push a real commit / pull request. Do not commit the dataset or invent commits for teammates.

## Dataset
Download Fruits-360 from Kaggle:
https://www.kaggle.com/datasets/moltean/fruits

Extract it so the structure looks like:
```text
data/fruits-360/
  Training/
    Apple Braeburn/
    Apple Golden 1/
    ...
  Test/
    Apple Braeburn/
    Apple Golden 1/
    ...
```
The training script uses the `Training` and `Test` folders and selects the configured classes that exist in both.

The dataset is not included in this repository because it is large and has its own license/terms. Review Kaggle's dataset terms before use.

## Setup
Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Train
After downloading and extracting the dataset:
```bash
python -m src.train --data-dir data/fruits-360 --output-dir artifacts --epochs 15
```

This creates:
- `artifacts/fruit_model.keras`
- `artifacts/class_names.json`
- `artifacts/training_history.json`
- `artifacts/metrics.json`
- `artifacts/confusion_matrix.png`

Copy `fruit_model.keras` and `class_names.json` into `models/` for the application.

## Run app
```bash
streamlit run app.py
```

## Streamlit application

The FruitScan AI web application is built using Streamlit.

The application provides the following functionality:
- Upload fruit and vegetable images.
- Display the uploaded image in the application.
- Send the image for classification using the trained CNN model.
- Display the predicted fruit or vegetable class.
- Show prediction confidence when available.

### Application interface

The interface is designed to be simple and user-friendly. Users can upload an image, start the classification process, and view the prediction results directly in their browser.

### Run the application

```bash
streamlit run app.py
```

The application will open in your default web browser.

### Participant 3 contribution

Participant 3 is responsible for:
- Developing the Streamlit web interface.
- Implementing image upload and preview.
- Displaying classification results.
- Updating project documentation and preparing application screenshots.

## Important limitations
- The model only recognizes the classes used during training.
- Fruits-360 images are typically isolated fruit/vegetable images on a plain background. Accuracy on casual phone photos may be lower.
- Softmax confidence is not a guarantee that the prediction is correct.
- This is an educational demonstration, not a food-safety or medical system.

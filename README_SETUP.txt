# Crop Recommendation Streamlit Prototype

## Files needed
Place these three files in the same folder:
- `app.py`
- `random_forest_model.pkl`
- `label_encoder.pkl`

The two `.pkl` files must be exported from your trained Random Forest model and its matching label encoder in Google Colab.

## 1. Export model files from Google Colab
Run this in your existing notebook, where `rf_model` and `label_encoder` are available:

```python
import joblib
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
```

Then download both files:

```python
from google.colab import files
files.download("random_forest_model.pkl")
files.download("label_encoder.pkl")
```

## 2. Install Python
Install Python on your computer. If loading the model fails due to a scikit-learn version mismatch, install the same scikit-learn version used in Colab.

## 3. Open Command Prompt in this folder
In File Explorer, open the folder containing `app.py`. Click the address bar, type `cmd`, and press Enter.

Optional: create a virtual environment:

```bat
python -m venv .venv
.venv\Scripts\activate
```

## 4. Install dependencies

```bat
python -m pip install -r requirements.txt
```

## 5. Start the app

```bat
python -m streamlit run app.py
```

Streamlit should open the app in your browser. If it does not, open the local URL shown in Command Prompt, usually `http://localhost:8501`.

To stop the app, click the Command Prompt window and press `Ctrl+C`.

## Important research notes
- The app sends seven entered values to the saved model: N, P, K, temperature, humidity, pH and rainfall.
- The location field is displayed for context only. It does not affect the prediction.
- Automatic weather-forecast retrieval is not included in this version.
- Rainfall must use a measurement period and units compatible with the training data.
- The output is a model prediction, not a guarantee of crop success. Validate recommendations with local agricultural advice and suitable field data.
- Only load `.pkl` files you created and trust. Pickle/joblib files from untrusted sources can execute code.

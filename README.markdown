# Water Quality Prediction

This project uses an Artificial Neural Network (ANN) to predict water potability based on parameters like pH, Hardness, etc. It includes a Streamlit web app for user interaction.

## Setup
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Download dataset**:
   - Get `water_potability.csv` from [Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability).
   - Place it in the `data/` folder.
3. **Download favicon**:
   - Get any `.ico` file (e.g., from [flaticon.com](https://www.flaticon.com/)) and place it in `static/images/favicon.ico`.
4. **Train the model**:
   ```bash
   python train_model.py
   ```
   This generates `water_quality_model.h5` and `scaler.pkl`.
5. **Run the web app**:
   ```bash
   streamlit run app.py
   ```
   Access the app at `http://localhost:8501`.

## Files
- `train_model.py`: Trains the ANN model.
- `app.py`: Streamlit web app for predictions.
- `utils.py`: Utility functions for input validation.
- `config.yaml`: Feature ranges for input validation.
- `static/css/style.css`: Custom CSS for the web app.
- `static/images/favicon.ico`: Favicon (not included).
- `requirements.txt`: Project dependencies.
- `data/water_potability.csv`: Dataset (not included).

## Notes
- The app outputs "Potability: Yes" or "Potability: No".
- Ensure the dataset and favicon are in the correct folders before running.
- For cloud deployment, push to GitHub and use Streamlit Cloud.
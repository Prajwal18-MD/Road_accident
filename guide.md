# User Guide - Road Accident Safety Prediction

Follow these instructions to set up and run the project locally.

## Prerequisite: Python 13.12.5
Ensure you have Python 13.12.5 installed. You can check your version with:
```bash
python --version
```

## 1. Create a Virtual Environment

Open your terminal (PowerShell or Command Prompt) in this directory and run:

```bash
# Create virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.\venv\Scripts\activate
```

## 2. Install Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

## 3. Train the Model

The training logic is in `train_notebook.py`. This is a Jupytext file, meaning it's a valid Python script that can also be opened as a Notebook in Jupyter.

**Option A: Run as a script**
```bash
python train_notebook.py
```

**Option B: Open in Jupyter Lab**
```bash
jupyter lab train_notebook.py
```
*Note: In Jupyter Lab, right-click the file and choose "Open With -> Notebook" if it opens as text.*

### Steps inside the Notebook:
1.  **Run all cells**.
2.  **Check Column Mapping**: If the notebook fails at the data loading stage, check the "Column Mapping" section. You may need to edit the `column_mapping` dictionary to match the headers in your `accident_prediction_india.csv`.
3.  **Wait for completion**: The script will output accuracy metrics and save a file named `model.joblib`.

## 4. Run the Streamlit App

Once `model.joblib` exists, launch the application:

```bash
streamlit run streamlit_app.py
```
This will open a new tab in your web browser where you can input accident details and get a risk prediction.

## Troubleshooting

-   **Missing Columns**: If you see a `KeyError` about missing columns (e.g., `'Time'`, `'Weather'`), open `train_notebook.py` and adjust the `column_mapping` dictionary to match your CSV's actual column names.
-   **Model Not Found**: If the Streamlit app says "Model file not found", ensure you have successfully run the training notebook/script first.

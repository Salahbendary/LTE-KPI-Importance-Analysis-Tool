import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button, Frame, Listbox, Scrollbar
from PIL import Image, ImageTk

# Define the file to store the results
results_file = 'kpi_hyperparameters.json'
data = None
selected_kpi = None

def save_kpi_results(kpi, hyperparameters, feature_importance_df):
    # Prepare the data to save
    result_data = {
        'hyperparameters': hyperparameters,
        'feature_importances': feature_importance_df.to_dict(orient='records')
    }
    
    # Save to a JSON file, appending if the file exists
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
    else:
        existing_results = {}

    existing_results[kpi] = result_data

    with open(results_file, 'w') as f:
        json.dump(existing_results, f, indent=4)


def load_kpi_results(kpi):
    # Load the stored results for the given KPI
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        return results.get(kpi, None)
    else:
        return None


def run_hyperparameter_tuning(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params 


def compute_feature_importance(X_train, y_train, best_params, features):
    best_model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        max_features=best_params['max_features'],
        min_samples_split=best_params['min_samples_split'],
        random_state=42
    )
    best_model.fit(X_train, y_train)
    feature_importances = best_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    return importance_df


def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx"), ("CSV Files", "*.csv")])
    if file_path:
        global data
        if file_path.endswith(".xlsx"):
            data = pd.read_excel(file_path)
        elif file_path.endswith(".csv"):
            data = pd.read_csv(file_path)
        messagebox.showinfo("Success", "File loaded successfully!")


def load_json():
    json_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    if json_path:
        global results_file
        results_file = json_path
        messagebox.showinfo("Success", "JSON file loaded successfully!")


def process_data():
    if data is None:
        messagebox.showerror("Error", "Please load a data file first!")
        return

    # Print column names to help debug
    print("Data Columns:", data.columns)

    # Check if the 'Date' and 'Time' columns exist
    if 'Date' in data.columns and 'Time' in data.columns:
        data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%m/%d/%Y %H:%M:%S.%f', errors='coerce')
        data.drop(['Time', 'Date'], axis=1, inplace=True)
    elif 'Datetime' in data.columns:
        data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
    else:
        messagebox.showerror("Error", "Missing 'Date' and 'Time' columns in the data!")
        return

    # Label encoding for categorical columns if present
    if 'PDSCH Modulation 1: Top #1' in data.columns:
        label_encoder = LabelEncoder()
        data['PDSCH Modulation 1: Top #1'] = label_encoder.fit_transform(data['PDSCH Modulation 1: Top #1'])

    # Select the KPI from the selection list
    if selected_kpi is None:
        messagebox.showerror("Error", "Please select a KPI first!")
        return

    print(f"Selected KPI: {selected_kpi}")

    # Split features and target variable
    features = [col for col in data.columns if col != selected_kpi and col != 'Datetime' and col != 'Unnamed: 0']
    X = data[features]
    y = data[selected_kpi]

    # Handle missing values
    imputer_X = SimpleImputer(strategy='median')
    X_imputed = imputer_X.fit_transform(X)
    imputer_y = SimpleImputer(strategy='median')
    y_imputed = imputer_y.fit_transform(y.values.reshape(-1, 1)).ravel()
    X = pd.DataFrame(X_imputed, columns=features)
    y = pd.Series(y_imputed, name=selected_kpi)

    # Print feature shapes to debug
    print("Features Shape:", X.shape)
    print("Target Shape:", y.shape)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load stored results if they exist
    stored_results = load_kpi_results(selected_kpi)
    if stored_results:
        best_params = stored_results['hyperparameters']
        feature_importance_df = pd.DataFrame(stored_results['feature_importances'])

        # Plot the feature importance
        top_5_features = feature_importance_df.head(5)
        plt.figure(figsize=(10, 6))
        plt.bar(top_5_features['Feature'], top_5_features['Importance'])
        plt.ylabel('KPIs Importance')
        plt.title(f'Top 5 KPIs Affecting {selected_kpi}')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.show()

    else:
        best_params = run_hyperparameter_tuning(X_train, y_train)
        feature_importance_df = compute_feature_importance(X_train, y_train, best_params, features)
        save_kpi_results(selected_kpi, best_params, feature_importance_df)

        print(f"Best KPIs for {selected_kpi}: {best_params}")
        print("KPIs Importance:")
        print(feature_importance_df)

        # Plot the feature importance
        top_5_features = feature_importance_df.head(5)
        plt.figure(figsize=(10, 6))
        plt.bar(top_5_features['Feature'], top_5_features['Importance'])
        plt.ylabel('KPIs Importance')
        plt.title(f'Top 5 KPIs Affecting {selected_kpi}')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.show()


# GUI Setup
root = tk.Tk()
root.title("KPI Importance Tool")
root.geometry("600x400")

# Add the logo image
logo_img = Image.open("vodafone_logo.png")  
logo_img = logo_img.resize((50, 50), Image.Resampling.LANCZOS)
logo_photo = ImageTk.PhotoImage(logo_img)
logo_label = Label(root, image=logo_photo)
logo_label.place(x=10, y=10)

# Title and creator info
title_label = Label(root, text="KPI Importance Tool", font=("Helvetica", 16, 'bold'))
title_label.pack(pady=20)
created_by_label = Label(root, text="Created by Salah Bendary", font=("Helvetica", 10))
created_by_label.pack(pady=5)

# Frame for buttons
frame = Frame(root)
frame.pack(pady=20)

# Button to upload data file
upload_button = Button(frame, text="Upload Data (CSV/Excel)", command=load_file)
upload_button.grid(row=0, column=0, padx=10)

# Button to upload JSON file
json_button = Button(frame, text="Upload JSON File", command=load_json)
json_button.grid(row=0, column=1, padx=10)

# Add a dropdown or listbox for KPI selection
kpi_label = Label(root, text="Select KPI:", font=("Helvetica", 10))
kpi_label.pack(pady=10)

kpi_listbox = Listbox(root, height=6, width=50)
kpi_listbox.pack(pady=5)
scrollbar = Scrollbar(root)
scrollbar.pack(side="right", fill="y")
kpi_listbox.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=kpi_listbox.yview)

def update_kpi_choice():
    global selected_kpi
    try:
        selected_kpi = kpi_listbox.get(kpi_listbox.curselection()[0])
    except IndexError:
        messagebox.showerror("Error", "Please select a KPI from the list.")

# Populate the listbox with available KPI columns after data load
def populate_kpi_list():
    if data is not None:
        kpi_columns = [col for col in data.columns if col not in ['Datetime', 'Unnamed: 0']]
        kpi_listbox.delete(0, tk.END)
        for col in kpi_columns:
            kpi_listbox.insert(tk.END, col)
    else:
        messagebox.showerror("Error", "Load data file first to populate KPIs.")

# Button to populate KPI list
populate_button = Button(root, text="Populate KPI List", command=populate_kpi_list)
populate_button.pack(pady=5)

# Button to start processing
process_button = Button(frame, text="Generate Results", command=lambda: [update_kpi_choice(), process_data()])
process_button.grid(row=1, columnspan=2, pady=20)

# Start the Tkinter loop
root.mainloop()

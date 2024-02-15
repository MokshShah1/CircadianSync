import streamlit as st
import pandas as pd
import os
import joblib

def main():
    with open("circsync_css.css", "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    
    st.markdown('<h1 class="centered-title">Circadian Sync</h1>', unsafe_allow_html=True)

    # Add content to the main area of the app
    st.markdown('<div class="main-content">CircadianSync is a Machine Learning model that intakes the gene expression levels of patients in order to analyze and predict whether they have pancreatic adenocarcinoma, circadian dysfunction, neither, or both.</div>', unsafe_allow_html=True)

    # Add widgets to the sidebar
    st.sidebar.title("CircSync Predictor")
    st.sidebar.markdown('<div class="sidebar-content">To use CircSync to predict a patient\'s diagnosis, upload a CSV/Excel file with their gene expression levels as numerical values in it. Make sure there are two columns: one for gene names labeled GeneID and one for the values labeled ExpressionLevels</div>', unsafe_allow_html=True)

    # Upload the CSV file
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(uploaded_file)
    
        # Remove the first column
        df.drop(df.columns[0], axis=1, inplace=True)
    
        # Move the second column to the first position
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]
    
        df.to_csv("modified_file.csv", index=False)

        # Load the test data from a file
        def load_test_data(file_path):
            df = pd.read_csv(file_path)  # assuming it's a CSV file
            return df

        # Preprocess the test data
        def preprocess_test_data(df):
            # Convert DataFrame values to numeric
            df = df.apply(pd.to_numeric, errors='coerce')  # coerce errors to NaN

            # Remove rows containing NaN values
            df.dropna(inplace=True)

            return df

        # Calculate gene expression ratios
        def calculate_ratios(df):
            # Calculate the sum of all cells in the file
            file_sum = df.values.sum()

            # Calculate ratios for each cell in the file
            file_ratios = df.values / file_sum

            # Flatten the nested list structure
            flattened_ratios = file_ratios.flatten()

            return flattened_ratios.reshape(1, -1)  # Reshape for prediction

        # Provide the folder path containing test files
        file_path = 'modified_file.csv'
        
        # Check if the file exists
        if os.path.exists(file_path) and file_path.endswith('.csv'):
            # Load and preprocess the test data
            test_data = load_test_data(file_path)
            test_data = preprocess_test_data(test_data)

            # Calculate gene expression ratios for the test data
            test_ratios = calculate_ratios(test_data)

            # Load the model from the file
            model = joblib.load('random_forest_model_ISEF (2).pkl')

            # Now you can use the loaded model to make predictions
            predictions = model.predict(test_ratios)

            # Print the predicted scenario for the file
            st.write(f"File: {file_path}, Predicted Scenario: {predictions}")

    # Dropdown menu in the sidebar
    selected_option = st.sidebar.selectbox("Select a model", ["Random Forest", "Gradient Boosting Classifier", "K Neighbors", "Decision Tree Classifier"])

    # Display selected option
    st.write(f"You selected: {selected_option}")

if __name__ == "__main__":
    main()

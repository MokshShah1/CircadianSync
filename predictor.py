import streamlit as st
import pandas as pd
import os
import joblib
import openpyxl  # Add this line to import openpyxl

def main():
    with open("circsync_css.css", "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    
    st.markdown('<h1 class="centered-title">Circadian Sync</h1>', unsafe_allow_html=True)

    # Add content to the main area of the app
    st.markdown('<div class="main-content">CircadianSync is a Machine Learning model that intakes the gene expression levels of patients in order to analyze and predict whether they have pancreatic adenocarcinoma, circadian dysfunction, neither, or both.</div>', unsafe_allow_html=True)

    # Add widgets to the sidebar
    st.sidebar.title("CircSync Predictor")
    st.sidebar.markdown('<div class="sidebar-content">To use CircSync to predict a patient\'s diagnosis, upload an Excel file with their gene expression levels as numerical values in it. Make sure there are two columns: one for gene names labeled GeneID and one for the values labeled ExpressionLevels</div>', unsafe_allow_html=True)

    # Upload the Excel file
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["xlsx"])

    if uploaded_file is not None:
        # Display the number of rows in the file
        df = pd.read_excel(uploaded_file, header=None)
        df.to_excel("imported_data.xlsx", index=False)  # Save to Excel file without index
        df.to_excel("modified_file.xlsx", index=False)

        # Load the test data from a file
        def load_test_data(file_path):
            df = pd.read_excel(file_path)  # assuming it's an Excel file
            return df

        # Preprocess the test data
        def preprocess_test_data(df):
            # Convert DataFrame values to numeric
            df = df.apply(pd.to_numeric, errors='coerce')  # coerce errors to NaN

            # Fill NaN values with zeros
            df.fillna(0, inplace=True)

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

        # Provide the file path for the modified file
        file_path = 'modified_file.xlsx'

        # Check if the file exists
        if os.path.exists(file_path) and file_path.endswith('.xlsx'):
            # Load and preprocess the test data
            test_data = load_test_data(file_path)
            test_data = preprocess_test_data(test_data)

            # Calculate gene expression ratios for the test data
            test_ratios = calculate_ratios(test_data)

            # Specify the path to the model file
            model_file_path = 'random_forest_model_ISEF (2).pkl'  # Replace 'path/to/' with the actual path to your model file

            # Check if the model file exists
            if os.path.exists(model_file_path):
                # Load the model from the file
                model = joblib.load(model_file_path)

                # Now you can use the loaded model to make predictions and obtain prediction probabilities
                predictions = model.predict(test_ratios)
                prediction_probabilities = model.predict_proba(test_ratios)

                # Mapping dictionary for transforming predicted scenarios
                prediction_labels = {
                    'pancreatic_circadian': 'Pancreatic Cancer and Disrupted Circadian Rhythm',
                    'no_pancreatic_circadian': 'No Pancreatic Cancer, but Disrupted Circadian Rhythm',
                    'no_pancreatic_no_circadian': 'No Pancreatic Cancer and Regular Circadian Rhythm',
                    'pancreatic_no_circadian': 'Pancreatic Cancer but Regular Circadian Rhythm'
                }

                # Transform predicted scenarios
                transformed_predictions = [prediction_labels[prediction] for prediction in predictions]

                # Print the transformed predicted scenario and prediction probabilities for the file
                st.write("Predicted Scenario:")
                for transformed_prediction in transformed_predictions:
                    st.write(transformed_prediction)

                # Show the prediction probabilities
                st.write("Prediction Probabilities:")
                for i, class_name in enumerate(model.classes_):
                    st.write(f"{class_name}: {prediction_probabilities[0][i]}")
            else:
                st.error(f"Model file '{model_file_path}' not found.")

    # Dropdown menu in the sidebar
    selected_option = st.sidebar.selectbox("Select a model", ["Random Forest", "Gradient Boosting Classifier", "K Neighbors", "Decision Tree Classifier"])

    # Display selected option
    st.write(f"You selected: {selected_option}")

if __name__ == "__main__":
    main()

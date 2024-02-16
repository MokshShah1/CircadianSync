import streamlit as st
import pandas as pd
import os
import joblib

def load_and_predict_model(test_ratios, selected_option):
    model_files = {
        "Random Forest": 'random_forest_model_ISEF (2).pkl',
        "Gradient Boosting Classifier": 'GradientBoostingClassifier_ISEF (2).pkl',
        "K Neighbors": 'KNeighborsClassifier_ISEF (2).pkl',
        "Decision Tree Classifier": 'DecisionTreeClassifier_ISEF (2).pkl'
    }
    
    model_file_path = model_files.get(selected_option)

    if model_file_path is not None and os.path.exists(model_file_path):
        model = joblib.load(model_file_path)
        predictions = model.predict(test_ratios)
        prediction_probabilities = model.predict_proba(test_ratios)

        prediction_labels = {
            'pancreatic_circadian': 'Pancreatic Cancer and Disrupted Circadian Rhythm',
            'no_pancreatic_circadian': 'No Pancreatic Cancer, but Disrupted Circadian Rhythm',
            'no_pancreatic_no_circadian': 'No Pancreatic Cancer and Regular Circadian Rhythm',
            'pancreatic_no_circadian': 'Pancreatic Cancer but Regular Circadian Rhythm'
        }

        transformed_predictions = [prediction_labels[prediction] for prediction in predictions]

        # Predicted Scenario Section
        st.markdown('<div class="predicted-scenario">', unsafe_allow_html=True)
        st.markdown('<h3>Predicted Scenario:</h3>', unsafe_allow_html=True)
        for transformed_prediction in transformed_predictions:
            st.markdown(f'<p>{transformed_prediction}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Prediction Probabilities Section
        st.markdown('<div class="prediction-probabilities">', unsafe_allow_html=True)
        st.markdown('<h3>Prediction Probabilities:</h3>', unsafe_allow_html=True)
        for i, class_name in enumerate(model.classes_):
            transformed_class_name = prediction_labels[class_name]
            st.markdown(f'<p>{transformed_class_name}: {prediction_probabilities[0][i]}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.error(f"Model file '{model_file_path}' not found.")

def main():
    # CSS styles
    st.markdown("""
        <style>
            /* Paste the CSS styles here */
            /* Global Styles */
            body {
                font-family: Arial, sans-serif;
                background-color: #f0f2f6; /* Light gray background */
                color: #333333; /* Dark text color */
                padding: 20px;
            }

            .centered-title {
                text-align: center;
                color: #4CAF50; /* Green title color */
                margin-bottom: 30px;
            }

            .main-content {
                margin-top: 20px;
                padding: 20px;
                background-color: #ffffff; /* White background for main content */
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Box shadow for a card-like effect */
            }

            .sidebar-content {
                padding: 20px;
                background-color: #ffffff; /* White background for sidebar content */
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Box shadow for a card-like effect */
            }

            .predicted-scenario,
            .prediction-probabilities {
                padding: 15px; /* Add padding for text elements */
                margin-bottom: 30px; /* Add spacing between text elements */
                background-color: #f0f0f0; /* Light gray background */
                border-radius: 5px; /* Rounded corners for text elements */
            }

            .predicted-scenario h3,
            .prediction-probabilities h3 {
                margin-bottom: 10px; /* Add spacing between titles and content */
            }

            .predicted-scenario p,
            .prediction-probabilities p {
                margin: 5px 0; /* Add spacing between lines of text */
            }

            .predict-button {
                background-color: #4CAF50; /* Green button background */
                color: white; /* White button text */
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                margin-top: 20px; /* Add spacing above the button */
            }

            .predict-button:hover {
                background-color: #45a049; /* Darker green on hover */
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="centered-title">Circadian Sync</h1>', unsafe_allow_html=True)

    st.markdown('<div class="main-content">CircadianSync is a Machine Learning model that intakes the gene expression levels of patients in order to analyze and predict whether they have pancreatic adenocarcinoma, circadian dysfunction, neither, or both.</div>', unsafe_allow_html=True)

    st.sidebar.title("CircSync Predictor")
    st.sidebar.markdown('<div class="sidebar-content">To use CircSync to predict a patient\'s diagnosis, upload an Excel file with their gene expression levels as numerical values in it. Make sure there are two columns: one for gene names labeled GeneID and one for the values labeled ExpressionLevels</div>', unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, header=None)
        df.to_excel("imported_data.xlsx", index=False)
        df.to_excel("modified_file.xlsx", index=False)

        def load_test_data(file_path):
            df = pd.read_excel(file_path)
            return df

        def preprocess_test_data(df):
            df = df.apply(pd.to_numeric, errors='coerce')
            df.fillna(0, inplace=True)
            return df

        def calculate_ratios(df):
            file_sum = df.values.sum()
            file_ratios = df.values / file_sum
            flattened_ratios = file_ratios.flatten()
            return flattened_ratios.reshape(1, -1)

        file_path = 'modified_file.xlsx'

        if os.path.exists(file_path) and file_path.endswith('.xlsx'):
            test_data = load_test_data(file_path)
            test_data = preprocess_test_data(test_data)
            test_ratios = calculate_ratios(test_data)

            selected_option = st.sidebar.selectbox("Select a model", ["Random Forest", "Gradient Boosting Classifier", "K Neighbors", "Decision Tree Classifier"])

            if selected_option:
                load_and_predict_model(test_ratios, selected_option)

if __name__ == "__main__":
    main()

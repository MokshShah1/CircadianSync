import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

# Function to create pie chart
def create_pie_chart(prediction_probabilities):
    labels = ['Pancreatic Cancer and Disrupted Circadian Rhythm',
              'No Pancreatic Cancer, but Disrupted Circadian Rhythm',
              'No Pancreatic Cancer and Regular Circadian Rhythm',
              'Pancreatic Cancer but Regular Circadian Rhythm']
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    fig1, ax1 = plt.subplots()
    ax1.pie(prediction_probabilities, colors=colors, labels=labels, autopct='%1.1f%%', startangle=140)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)

# Load and predict model function
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

        st.markdown('<div class="predicted-scenario">Predicted Scenario:</div>', unsafe_allow_html=True)
        for i, class_name in enumerate(model.classes_):
            transformed_class_name = prediction_labels[class_name]
            st.write(f"{transformed_class_name}: {prediction_probabilities[0][i]}")

        # Create and display pie chart
        create_pie_chart(prediction_probabilities[0])

    else:
        st.error(f"Model file '{model_file_path}' not found.")

# Main function
def main():
    st.markdown('<style>body {background-color: #121212; color: #ffffff;}</style>', unsafe_allow_html=True)  # Set background color and text color

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

import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

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

        prediction_labels = {
            'pancreatic_circadian': 'Pancreatic Cancer and Disrupted Circadian Rhythm',
            'no_pancreatic_circadian': 'No Pancreatic Cancer, but Disrupted Circadian Rhythm',
            'no_pancreatic_no_circadian': 'No Pancreatic Cancer and Regular Circadian Rhythm',
            'pancreatic_no_circadian': 'Pancreatic Cancer but Regular Circadian Rhythm'
        }

        st.markdown('**Predicted Scenario:**')
        for i, class_name in enumerate(model.classes_):
            transformed_class_name = prediction_labels[class_name]
            st.markdown(f'**{transformed_class_name}:** {prediction_probabilities[0][i]}')

        # Create and display pie chart
        fig, ax1 = plt.subplots()
        ax1.pie(prediction_probabilities[0], labels=prediction_labels.values(), autopct='%1.1f%%', startangle=140)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

    else:
        st.error(f"Model file '{model_file_path}' not found.")

# Main function
def main():
    st.title("Circadian Sync")

    st.write("CircadianSync is a Machine Learning model that intakes the gene expression levels of patients in order to analyze and predict whether they have pancreatic adenocarcinoma, circadian dysfunction, neither, or both.")

    st.sidebar.title("CircSync Predictor")
    st.sidebar.write("To use CircSync to predict a patient's diagnosis, upload an Excel file with their gene expression levels as numerical values in it. Make sure there are two columns: one for gene names labeled GeneID and one for the values labeled ExpressionLevels")

    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, header=None)
        df.to_excel("imported_data.xlsx", index=False)
        df.to_excel("modified_file.xlsx", index=False)

        file_path = 'modified_file.xlsx'

        if os.path.exists(file_path) and file_path.endswith('.xlsx'):
            test_data = pd.read_excel(file_path)
            test_data.fillna(0, inplace=True)
            test_ratios = test_data.sum().values / test_data.sum().sum()

            selected_option = st.sidebar.selectbox("Select a model", ["Random Forest", "Gradient Boosting Classifier", "K Neighbors", "Decision Tree Classifier"])

            if selected_option:
                load_and_predict_model(test_ratios.reshape(1, -1), selected_option)

if __name__ == "__main__":
    main()

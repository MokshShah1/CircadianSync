import streamlit as st
import pandas as pd
import os
import joblib
import plotly.graph_objects as go

def create_pie_chart(prediction_probabilities):
    labels = [
        'Pancreatic Cancer and Disrupted Circadian Rhythm',
        'No Pancreatic Cancer and Regular Circadian Rhythm',
        'No Pancreatic Cancer, but Disrupted Circadian Rhythm',
        'Pancreatic Cancer but Regular Circadian Rhythm'
    ]
    
    # Filter out probabilities that are 0
    filtered_probabilities = [p for p in prediction_probabilities if p != 0]
    filtered_labels = [labels[i] for i, p in enumerate(prediction_probabilities) if p != 0]
    
    fig = go.Figure(data=[go.Pie(
        labels=filtered_labels,
        values=filtered_probabilities,
        hoverinfo='label+percent',
        textinfo='percent',  # Changed to show only percentage
        textfont=dict(size=20, color='white', family='Arial'),  # Made text bigger
        marker=dict(colors=['#6a5acd', '#4682b4', '#00bcd4', '#ff8c00']),
        pull=[0.1 if p == max(filtered_probabilities) else 0 for p in filtered_probabilities],
        rotation=90
    )])
    
    # Adjust layout to center title and improve appearance
    fig.update_layout(
        title="Prediction Results",
        title_x=0.36,  # Ensures the title is centered
        title_font_size=24,
        template='plotly_dark',
        showlegend=False,
        margin=dict(t=50, b=50, l=50, r=50),
        height=400,
        width=500,
        xaxis=dict(showgrid=False, zeroline=False),  # Ensure no grid on the x-axis
        yaxis=dict(showgrid=False, zeroline=False)   # Ensure no grid on the y-axis
    )

    # Calculate max probability and corresponding label for display
    max_probability = max(filtered_probabilities)
    max_label = filtered_labels[filtered_probabilities.index(max_probability)]
    
    return fig, filtered_labels, filtered_probabilities, max_label, max_probability

def load_and_predict_model(test_ratios, selected_option):
    model_files = {
        "Random Forest": 'random_forest_model_ISEF (2).pkl',
        "Gradient Boosting Classifier": 'GradientBoostingClassifier_ISEF (2).pkl',
        "K Neighbors": 'KNeighborsClassifier_ISEF (2).pkl',
        "Decision Tree Classifier": 'DecisionTreeClassifier_ISEF (2).pkl'
    }

    if selected_option == "None":
        st.warning("No model selected. Please choose a model to proceed.")
        return  # Do nothing if "None" is selected

    model_file_path = model_files.get(selected_option)

    if os.path.exists(model_file_path):
        model = joblib.load(model_file_path)
        prediction_probabilities = model.predict_proba(test_ratios)

        # Generate the pie chart and get all labels and probabilities
        fig, filtered_labels, filtered_probabilities, max_label, max_probability = create_pie_chart(prediction_probabilities[0])

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Display all predictions with probabilities in a styled and centered way
        st.markdown('<h3 style="text-align: center; font-size: 30px; font-weight: bold; color: #ff8c00;">All Predictions with Probabilities:</h3>', unsafe_allow_html=True)
        for label, probability in zip(filtered_labels, filtered_probabilities):
            st.markdown(f"<p style='text-align: center; font-size: 24px; color: white;'>{label}: {probability * 100:.2f}%</p>", unsafe_allow_html=True)
        
        # Add space before displaying the max prediction
        st.markdown("<br><br>", unsafe_allow_html=True)  # Add extra space above the max prediction

        # Display the main prediction in two lines
        st.markdown('<h3 style="font-weight: bold; font-size: 30px; color: #ff8c00; text-align: center;">Main Prediction:</h3>', unsafe_allow_html=True)  # Orange text is larger and bold
        st.markdown(f'<h3 style="font-size: 24px; color: white; text-align: center; font-weight: normal;">{max_label}</h3>', unsafe_allow_html=True)  # White text, no probability shown
    else:
        st.error(f"Model file '{model_file_path}' not found.")

def main():
    st.markdown("""
    <style>
        body { font-family: Arial, sans-serif; background-color: #2f2f2f; color: #ffffff; padding: 20px; }
        .container { display: flex; flex-direction: column; align-items: center; gap: 30px; margin-top: 50px; }
        .centered-title { text-align: center; margin-bottom: 20px; color: #ff8c00; font-size: 36px; font-weight: bold; }
        .centered-text { text-align: center; font-size: 24px; color: white; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="centered-title">Circadian Sync</h1>', unsafe_allow_html=True)

    st.sidebar.title("CircSync Predictor")

    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["xlsx"])

    # Initialize selected_option to None
    selected_option = None

    if uploaded_file:
        df = pd.read_excel(uploaded_file, header=None)

        # Preprocess and calculate ratios
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        test_ratios = (df.values / df.values.sum()).flatten().reshape(1, -1)

        # Ensure the model selection prompt appears if no model is selected
        selected_option = st.sidebar.selectbox("Select a model", ["None", "Random Forest", "Gradient Boosting Classifier", "K Neighbors", "Decision Tree Classifier"])
        
        if selected_option != "None":
            load_and_predict_model(test_ratios, selected_option)

if __name__ == "__main__":
    main()

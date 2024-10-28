import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# Apply custom CSS (Optional, adjust "style.css" if you want custom styling)
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("StudentPerformanceFactors.csv")
    return data

data = load_data()

# Add a 'Grade' column based on the Exam_Score threshold
data['Grade'] = data['Exam_Score'].apply(lambda x: 'Pass' if x >= 65 else 'Fail')

# Sidebar for navigation
st.sidebar.title("Student Performance Prediction")
page = st.sidebar.selectbox("Select a page", ["Home", "Data Visualization", "Prediction"])

# Home page
if page == "Home":
    st.title("Welcome to the Student Performance Prediction App")
    st.image("image1.png", use_column_width=True)
    st.write(
        """
        This app predicts student performance based on various factors. 
        Use the Data Visualization page to explore insights in the dataset,
        or head to the Prediction page to make predictions based on your inputs.
        """
    )

# Data Visualization page
elif page == "Data Visualization":
    st.title("Data Visualization")
    st.write("Select features to visualize their relationships with student performance.")
    
    if st.checkbox("Show Correlation Heatmap"):
        st.write("Correlation Heatmap of Features")
        numeric_data = data.select_dtypes(include=['float64', 'int64']).dropna()
        
        if not numeric_data.empty:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            st.pyplot(plt)
            plt.clf()

    chart_type = st.selectbox("Select chart type", ["Histogram", "Scatter", "Box"])

    if chart_type == "Histogram":
        feature = st.selectbox("Select feature for histogram", data.columns)
        plt.figure(figsize=(10, 6))
        sns.histplot(data[feature], kde=True, color='blue')
        st.pyplot(plt)
    else:
        x_var = st.selectbox("Select X-axis variable", data.columns)
        y_var = st.selectbox("Select Y-axis variable", data.columns)
        
        plt.figure(figsize=(10, 6))
        if chart_type == "Scatter":
            sns.scatterplot(x=data[x_var], y=data[y_var])
        elif chart_type == "Box":
            sns.boxplot(x=x_var, y=y_var, data=data)
        st.pyplot(plt)

# Prediction page
elif page == "Prediction":
    st.title("Predict Student Exam Score and Grade")

    features = {
        "Hours_Studied": st.number_input("Enter Hours Studied per Week", min_value=0, max_value=50, step=1),
        "Attendance": st.number_input("Enter Attendance Percentage", min_value=0, max_value=100, step=1),
        "Previous_Scores": st.number_input("Enter Previous Scores (0-100)", min_value=0, max_value=100, step=1),
        "Tutoring_Sessions": st.number_input("Enter Number of Tutoring Sessions per Week", min_value=0, max_value=6, step=1),
        "Sleep_Hours": st.number_input("Enter Average Sleep Hours per Night", min_value=0, max_value=15, step=1),
        "Physical_Activity": st.number_input("Enter Physical Activities per Week", min_value=0, max_value=5, step=1)
    }
    
    input_df = pd.DataFrame([features])

    @st.cache_resource
    def train_regressor(X_train, y_train):
        regressor = RandomForestRegressor(random_state=42)
        regressor.fit(X_train, y_train)
        return regressor

    @st.cache_resource
    def train_classifier(X_train, y_train):
        classifier = RandomForestClassifier(random_state=42)
        classifier.fit(X_train, y_train)
        return classifier

    if st.button("Predict"):
        X = data[["Hours_Studied", "Attendance", "Previous_Scores", 
                  "Tutoring_Sessions", "Sleep_Hours", "Physical_Activity"]]
        y_score = data["Exam_Score"]
        y_grade = data["Grade"]

        X_train, X_test, y_score_train, y_score_test = train_test_split(X, y_score, test_size=0.2, random_state=42)
        _, _, y_grade_train, y_grade_test = train_test_split(X, y_grade, test_size=0.2, random_state=42)

        regressor = train_regressor(X_train, y_score_train)
        classifier = train_classifier(X_train, y_grade_train)

        predicted_score = regressor.predict(input_df)[0]
        predicted_grade = classifier.predict(input_df)[0]

        y_score_pred = regressor.predict(X_test)
        y_grade_pred = classifier.predict(X_test)

        mse = mean_squared_error(y_score_test, y_score_pred)
        accuracy = accuracy_score(y_grade_test, y_grade_pred)

        st.write(f"Model Mean Squared Error for Exam Score Prediction: {mse:.2f}")
        st.write(f"Model Accuracy for Grade Prediction: {accuracy * 100:.2f}%")

        st.write(f"Predicted Exam Score: {predicted_score:.2f}")

        # Highlight Grade with color
        if predicted_grade == 'Pass':
            st.markdown(f"<h3 style='color: green;'>Predicted Grade: {predicted_grade}</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color: red;'>Predicted Grade: {predicted_grade}</h3>", unsafe_allow_html=True)

        

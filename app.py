import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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

# Sidebar for navigation
st.sidebar.title("Student Performance Prediction")
page = st.sidebar.selectbox("Select a page", ["Home", "Data Visualization", "Prediction"])

# Home page
if page == "Home":
    st.title("Welcome to the Student Performance Prediction App")
    
    # Load image from a local path (adjust path if needed)
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
    
    # Check for numeric columns and missing values
    if st.checkbox("Show Correlation Heatmap"):
        st.write("Correlation Heatmap of Features")
        
        # Select only numerical columns and handle missing values
        numeric_data = data.select_dtypes(include=['float64', 'int64']).dropna()
        
        if numeric_data.empty:
            st.write("No numerical data available for the heatmap.")
        else:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            st.pyplot(plt)
            plt.clf()  # Clear the figure to avoid any overlap issues

    
    # Selecting visualization type
    chart_type = st.selectbox("Select chart type", ["Histogram", "Scatter", "Box"])

    # Generate histogram if selected
    if chart_type == "Histogram":
        feature = st.selectbox("Select feature for histogram", data.columns)
        plt.figure(figsize=(10, 6))
        sns.histplot(data[feature], kde=True, color='blue')
        st.pyplot(plt)
    
    # Scatter plot and box plot options
    else:
        x_var = st.selectbox("Select X-axis variable", data.columns)
        y_var = st.selectbox("Select Y-axis variable", data.columns)

        if chart_type == "Scatter":
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=data[x_var], y=data[y_var])
        elif chart_type == "Box":
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=x_var, y=y_var, data=data)
        
        st.pyplot(plt)

# Prediction page
elif page == "Prediction":
    st.title("Predict Student Exam Score")

    # Specific prediction features
    features = {
        "Hours_Studied": st.number_input("Enter Hours Studied per Week", min_value=0, max_value=50, step=1),
        "Attendance": st.number_input("Enter Attendance Percentage", min_value=0, max_value=100, step=1),
        "Previous_Scores": st.number_input("Enter Previous Scores (0-100)", min_value=0, max_value=100, step=1),
        "Tutoring_Sessions": st.number_input("Enter Number of Tutoring Sessions per Week", min_value=0,max_value=6, step=1),
        "Sleep_Hours": st.number_input("Enter Average Sleep Hours per Night", min_value=0, max_value=15, step=1),
        "Physical_Activity": st.number_input("Enter Physical Activities per Week", min_value=0, max_value=5, step=1)
    }
    
    # Convert input data into a DataFrame
    input_df = pd.DataFrame([features])

    # Model training and prediction function
    @st.cache_resource
    def train_model(X_train, y_train):
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        return model

    # Model training and prediction
    if st.button("Predict"):
        # Selecting features for model training and the target column
        X = data[["Hours_Studied", "Attendance", "Previous_Scores", 
                  "Tutoring_Sessions", "Sleep_Hours", "Physical_Activity"]]
        y = data["Exam_Score"]  # Using 'Exam_Score' as the target column

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train and cache the model
        model = train_model(X_train, y_train)

        # Prediction
        prediction = model.predict(input_df)
        st.write(f"Predicted Exam Score: {prediction[0]:.2f}")

        # Display model accuracy (using mean squared error as a metric)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Model Mean Squared Error: {mse:.2f}")

        # Feature importance
        #st.write("Feature Importance")
        #feature_importance = model.feature_importances_
        #importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importance})
        #importance_df = importance_df.sort_values(by="Importance", ascending=False)

        #plt.figure(figsize=(10, 6))
        #sns.barplot(x="Importance", y="Feature", data=importance_df)
        #st.pyplot(plt)

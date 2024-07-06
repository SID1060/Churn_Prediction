import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Churn Prediction", page_icon=":bar_chart:", layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('Churn_Modelling.csv')  # Ensure this file is in the same directory as app.py
    return data

data = load_data()

# Preprocess the data
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
y = data['Exited']
x = data.drop('Exited', axis=1)
x = pd.get_dummies(x, columns=['Geography', 'Gender'], drop_first=True)  # Encode categorical variables

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)  # Fit on training data and transform
x_test_scaled = scaler.transform(x_test)  # Transform test data

# Train the model
model = DecisionTreeClassifier()
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  # Required
        options=["Home", "EDA", "Model Performance", "Predict Churn"],  # Required
        icons=["house", "bar-chart", "bar-chart", "person-fill"],  # Optional
        menu_icon="cast",  # Optional
        default_index=0,  # Optional
    )

# Main content based on menu selection
if selected == "Home":
    st.title("Churn Prediction Model")
    st.write("""
    ## Overview
    This app predicts customer churn based on various features.
    """)


    # Icons for GitHub, LinkedIn, and Email
    st.markdown("""
    <div style="position: absolute; bottom: 20px; right: 20px;">
        <a href="https://github.com/SID1060/" target="_blank" style="margin-right: 10px;">
            <img src="https://img.icons8.com/ios-glyphs/30/000000/github.png"/>
        </a>
        <a href="https://www.linkedin.com/in/siddharth-verma-444673278/" target="_blank" style="margin-right: 10px;">
            <img src="https://img.icons8.com/ios-glyphs/30/000000/linkedin.png"/>
        </a>
        <a href="mailto:siddharth.verma@iitgn.ac.in" style="margin-right: 10px;">
            <img src="https://img.icons8.com/ios-glyphs/30/000000/new-post.png"/>
        </a>
    </div>
    """, unsafe_allow_html=True)

elif selected == "EDA":
    st.title("Exploratory Data Analysis")
    
    # Distribution of Age
    st.write("### Distribution of Age")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['Age'], bins=20, kde=True, ax=ax)
    ax.set_title('Distribution of Age')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    # Distribution of Geography
    st.write("### Distribution by Geography")
    fig, ax = plt.subplots()
    data.groupby('Geography').size().plot(kind='barh', color=sns.color_palette('Dark2'), ax=ax)
    ax.set_title('Geography Distribution')
    st.pyplot(fig)

    # Distribution of Gender
    st.write("### Distribution by Gender")
    fig, ax = plt.subplots()
    data.groupby('Gender').size().plot(kind='barh', color=sns.color_palette('Dark2'), ax=ax)
    ax.set_title('Gender Distribution')
    st.pyplot(fig)

    # Exited vs Retained
    st.write("### Exited vs Retained")
    fig, ax = plt.subplots()
    labels = ['Exited', 'Retained']
    sizes = [data.Exited[data['Exited'] == 1].count(), data.Exited[data['Exited'] == 0].count()]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
    st.pyplot(fig)

    # Distribution of Balance
    st.write("### Distribution of Balance")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['Balance'], bins=20, kde=True, ax=ax)
    ax.set_title('Distribution of Balance')
    ax.set_xlabel('Balance')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Exited or not from different perspectives
    st.write("### Exited or Not from Different Perspectives")
    fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
    sns.countplot(x='Geography', hue='Exited', data=data, ax=axarr[0][0])
    sns.countplot(x='Gender', hue='Exited', data=data, ax=axarr[0][1])
    sns.countplot(x='HasCrCard', hue='Exited', data=data, ax=axarr[1][0])
    sns.countplot(x='IsActiveMember', hue='Exited', data=data, ax=axarr[1][1])
    st.pyplot(fig)
    
    # Exited based on age
    st.write("### Exited Based on Age")
    fig = sns.pairplot(data[['Age', 'Exited']], hue='Exited')
    st.pyplot(fig)

    # Number of Products Distribution
    st.write("### Number of Products Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='NumOfProducts', data=data, ax=ax)
    ax.set_title('Number of Products Distribution')
    ax.set_xlabel('Number of Products')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # Tenure Distribution
    st.write("### Tenure Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Tenure', data=data, ax=ax)
    ax.set_title('Tenure Distribution')
    ax.set_xlabel('Tenure (years)')
    ax.set_ylabel('Count')
    st.pyplot(fig)

elif selected == "Model Performance":
    st.title("Model Performance Metrics")

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Display metrics
    st.markdown(f"**Accuracy:** {accuracy:.2f}")
    st.markdown(f"**Precision:** {precision:.2f}")
    st.markdown(f"**Recall:** {recall:.2f}")
    st.markdown(f"**F1-Score:** {f1:.2f}")

    # Confusion matrix
    st.write("### Confusion Matrix")
    st.write(conf_matrix)

elif selected == "Predict Churn":
    st.title("Predict Churn for a New Customer")

    # User input for prediction
    st.write("""
    Fill in the details below to predict whether a customer is likely to churn.
    """)
    age = st.slider("Age", min_value=18, max_value=100, value=30)
    balance = st.number_input("Balance", min_value=0, value=10000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
    estimated_salary = st.number_input("Estimated Salary", min_value=0, value=50000)
    tenure = st.slider("Tenure", min_value=0, max_value=10, value=5)

    # Mapping user input to dummy columns
    geography_france = 1  # Default assumption
    geography_germany = 0
    geography_spain = 0

    selected_geography = st.selectbox("Country", ['France', 'Germany', 'Spain'])
    if selected_geography == 'France':
        geography_france = 1
        geography_germany = 0
        geography_spain = 0
    elif selected_geography == 'Germany':
        geography_france = 0
        geography_germany = 1
        geography_spain = 0
    elif selected_geography == 'Spain':
        geography_france = 0
        geography_germany = 0
        geography_spain = 1

    # Gender selection
    gender = st.radio("Gender", ['Male', 'Female'])
    gender_male = 1 if gender == 'Male' else 0

    # Create new data frame for prediction
    new_data = pd.DataFrame({
        'Age': [age],
        'Balance': [balance],
        'CreditScore': [credit_score],
        'EstimatedSalary': [estimated_salary],
        'Tenure': [tenure],
        'Geography_France': [geography_france],
        'Geography_Germany': [geography_germany],
        'Geography_Spain': [geography_spain],
        'Gender_Male': [gender_male]
    })

    # Ensure all columns are present and in the same order as during training
    new_data = new_data.reindex(columns=x.columns, fill_value=0)

    # Scale the input features
    scaled_data = scaler.transform(new_data)

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(scaled_data)
        if prediction[0] == 1:
            st.error("The model predicts that the customer is likely to switch services.")
        else:
            st.success("The model predicts that the customer is unlikely to switch services.")

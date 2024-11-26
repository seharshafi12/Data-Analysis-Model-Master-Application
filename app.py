import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns


# Title of the app  
st.title(" Data Analysis Model Master Application")



# add a side bar
st.sidebar.header("Welcome to the Data Analysis App by SEHAR SHAFI")

# add a review box in side bar
st.sidebar.header("project description") 
st.sidebar.write("Welcome to our model master App! Our app empowers you to explore your data with seamless preprocessing and advanced modeling capabilities. Whether you're tackling classification or regression tasks, ModelMaster ensures your data is clean, structured, and ready for insightful predictions")
# add a check box in side bar  having options  good noemal bad exelent app
st.sidebar.header("Rate this app")
st.sidebar.write("Rate this app")
rating = st.sidebar.radio("Rate this app", ("Excelent","Good", "Normal", "Bad",))


# data selection
data_list = ["00", "iris", "tips","titanic", "diamonds", "anscombe"]
data_name = st.selectbox("Select data", data_list )


# load the selected data
if data_name == "iris":
    df = sns.load_dataset("iris")
elif data_name == "tips":
    df = sns.load_dataset("tips")
elif data_name == "titanic":
    df = sns.load_dataset("titanic")
elif data_name == "diamonds":
    df = sns.load_dataset("diamonds")
elif data_name == "anscombe":
    df = sns.load_dataset("anscombe")
else:
    st.write("No data found")


# add button to upload your own data in the siede bar
st.subheader("Upload your own data")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)


#  to check the head of the data
st.header("Data Preview")
st.write(df.head())

# number of rows and columns
st.header("Number of Rows and Columns")
st.write("Number of Rows:", df.shape[0])
st.write("Number of Columns:", df.shape[1])




# data shape 
st.header("Data Shape")
st.write(df.shape)




# display the columns names and their  data types
st.header("Data Types")
st.write(df.dtypes)



# describe the data
st.header("Describe the data")
st.write(df.describe())



# NUll values button

st.header("NUll values")
# add a button to check null values

if df.isnull().values.any():
        st.write(df.isnull().sum().sort_values(ascending=False))
    #select the coloum to remove the columns
        st.header("Drop columns")
        remove_columns = st.multiselect("Select columns to remove", df.columns, default=[])
        st.header("After removing columns")
        df = df.drop(remove_columns, axis=1)
        df.isnull().values.any()
        st.write(df.isnull().sum().sort_values(ascending=False))

        # select the colum to drop nan values
        st.header("Drop rows with missing values")
        drop_missing_values = st.multiselect("Select columns to drop", df.columns, default=[])
        df = df.dropna(subset=drop_missing_values)
        # show only selected coloumns which are dropped nan values
        st.write(df[drop_missing_values].isnull().sum().sort_values(ascending=False))
        

        # impute with mean
        st.header("Impute with mean")
        # select the one coloum to impute and show only numeric coloums
        impute_columns = st.multiselect("Select columns to impute by mean", df.select_dtypes(include=[np.number]).columns, default=[])
        # impute the coloum with mean
        df[impute_columns] = df[impute_columns].fillna(df[impute_columns].mean())
        # show only that coloumns which are imputed
        st.write(df[impute_columns].isnull().sum().sort_values(ascending=False))

        # impute the coloum with median
        st.header("Impute with median")
        # select the one coloum to impute which are numeric
        imputed_columns = st.multiselect("Select columns to impute by median", df.select_dtypes(include=[np.number]).columns, default=[])
        # show only that coloumns which are imputed
        st.write(df[imputed_columns].isnull().sum().sort_values(ascending=False))
        
        # impute the coloum with mode
        st.header("Impute with mode")
        # select the one coloum to impute select only categorical coloumns
        impute_columns = st.multiselect("Select columns to impute by mode", df.select_dtypes(include=[object]).columns, default=[])

        # show only that coloumns which are imputed
        st.write(df[impute_columns].isnull().sum().sort_values(ascending=False))
        


        # check null values and show them
        st.header("after imputing missing values")
        df.isnull().values.any()
        st.write(df.isnull().sum().sort_values(ascending=False))  
else:
        st.write("No Null values found") 





# add a button for the pair plot

st.header("Pair Plot")

# Select the column for adding hue in the pair plot
hue_color = st.selectbox("Select hue", df.columns)

if st.button("Show Pair Plot"):
    # Clear the current figure
    plt.clf()

    # Create the pair plot with the selected hue
    pair_plot = sns.pairplot(df, hue=hue_color)

    # Show the plot in Streamlit
    st.pyplot(pair_plot.fig)

   
# add a button for the heatmap
st.header("Heatmap")
if st.button("Show Heatmap"):
    # Clear the current figure
    plt.clf()
    # select the colums which are numeric for correlation
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_columns].corr()

    # Create the heatmap    
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")

    # Show the plot in Streamlit
    st.pyplot(heatmap.get_figure())

# check unique values in each coloumn
st.header("Unique Values")
for col in df.columns:
    st.write(f"{col} has {df[col].nunique()} unique values")

# values counts and made a sekect box for selecting coloums
st.header("Value Counts")
coloumns = st.multiselect("Select coloumns", df.columns)
for col in coloumns:
    st.write(df[col].value_counts())


# downlod the clean data
st.header("Download Clean Data")
st.download_button("Download", df.to_csv(index=False).encode('utf-8'), file_name="clean_data.csv")


# now we are done with data cleaning
st.header("Now we are done with data cleaning")

# lets start model building and prediction of classification or regression problem

# import the required libraries 
from sklearn.model_selection import train_test_split    
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,root_mean_squared_error


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve,classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score,precision_score,recall_score,f1_score
# import scaler and encoders 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


st.header("Model Building")

# select the x and y
st.subheader("Select the x and y variables")
x = st.multiselect("Select the x", df.columns)
y = st.selectbox("Select the y", df.columns)

# ask the user by adding a slider to split the data
st.subheader("Split the data")
test_size = st.slider("Test size", 0.1, 0.9, 0.2)
X = df[x]
y = df[y]

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# show the split data 
st.write("X_train shape:", X_train.shape)
st.write("X_test shape:", X_test.shape)
st.write("y_train shape:", y_train.shape)
st.write("y_test shape:", y_test.shape)


# scale or encode the data according to their type
st.subheader("Scale or encode the X data")
# ask the user to select the scaling method
scaling_method = st.selectbox("Select the scaling method", ["StandardScaler", "MinMaxScaler"])


if scaling_method == "StandardScaler":
    scaler = StandardScaler()
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
        X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
    else:
        st.write("No numeric columns found. Please select columns with numeric type.")
elif scaling_method == "MinMaxScaler":
    scaler = MinMaxScaler()
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
        X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
    else:
        st.write("No numeric columns found. Please select columns with numeric type.")


# Encoding
encoding_method = st.selectbox("Select the encoding method", ["none","OneHotEncoder", "LabelEncoder"])

if encoding_method == "OneHotEncoder":
    encoder = OneHotEncoder()
    categorical_columns = X_train.select_dtypes(include=['object','category']).columns
    if len(categorical_columns) > 0:
        X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
        X_test_encoded = encoder.transform(X_test[categorical_columns])
        X_train = pd.concat([X_train.drop(columns=categorical_columns), pd.DataFrame(X_train_encoded, index=X_train.index)], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_columns), pd.DataFrame(X_test_encoded, index=X_test.index)], axis=1)
    else:
        st.write("No categorical columns found. Please select columns with categorical type.")
elif encoding_method == "LabelEncoder":
    encoder = LabelEncoder()
    categorical_columns = X_train.select_dtypes(include=['object','category']).columns
    if len(categorical_columns) > 0:
        for col in categorical_columns:
            X_train[col] = encoder.fit_transform(X_train[col])
            X_test[col] = encoder.transform(X_test[col])
    else:
        st.write("No categorical columns found. Please select columns with categorical type.")


# Show the data after scaling and encoding
st.write(X_train.head())
st.write("Processed Test Data:")
st.write(X_test.head())
st.write("Please select at least one feature for X.")


# scale or encode the y if needed
st.subheader("Scale or encode the y")
st.write("we do not need to scale or encode the y,but encocode only when the y is obtect type or string type ,it cause error in traning a model when y type is string")
y_encoding = st.selectbox("Select the scaling or encoding ", ["none","scaling", "encoding"])

if y_encoding == "scaling":
    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test = scaler.transform(y_test.values.reshape(-1, 1))
elif y_encoding == "encoding":
    encoder = LabelEncoder()    
    y_train = encoder.fit_transform(y_train)    
    y_test = encoder.transform(y_test)

# Show the data after scaling and encoding
st.write(y_train)




problem_type = st.selectbox("Select the problem type", ["Regression", "Classification"])

if problem_type == "Regression":
    st.subheader("Select the model")
    model = st.selectbox("Select the model", ["Linear Regression", "Random Forest", "Decision Tree", 
    "Support Vector Regression", "K-Nearest Neighbors", "Gradient Boosting"])
    
    
    if model == "Linear Regression":
            st.subheader("Linear Regression")
            model = LinearRegression()
    elif model == "Random Forest":
            st.subheader("Random Forest")
            model = RandomForestRegressor()
    elif model == "Decision Tree":    
            st.subheader("Decision Tree")
            model = DecisionTreeRegressor()
    elif model == "Support Vector Regression":
            st.subheader("Support Vector Regression")
            model = SVR()
    elif model == "K-Nearest Neighbors":    
            st.subheader("K-Nearest Neighbors")    
            model = KNeighborsRegressor()    
    elif model == "Gradient Boosting":
            st.subheader("Gradient Boosting")    
            model = GradientBoostingRegressor() 
        

    # Training the model
    st.subheader("Train the model")
    model.fit(X_train, y_train)
    st.write("Model is trained now")

    # Making predictions
    if st.button("Make predictions"):
            y_pred = model.predict(X_test)
            st.write("Predictions:", y_pred)
        
            # Evaluate the model
    if st.button("Evaluate the model"):
            y_test_pred =model.predict(X_test)
            mse = mean_squared_error(y_test, y_test_pred)
            st.write("Mean Squared Error:", mse)
            r2 = r2_score(y_test, y_test_pred)
            st.write("R-squared:", r2)  
            rmse= root_mean_squared_error(y_test, y_test_pred)
            st.write("Root Mean Squared Error:", rmse)
            mae= mean_absolute_error(y_test, y_test_pred)
            st.write("Mean Absolute Error:", mae)

    else:
        st.write("Train the model first to make predictions or evaluate.")


    # made the plot for the model
    st.header("Plot the model")
    y_test_pred =model.predict(X_test)
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_test_pred)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted Values")
    st.pyplot(fig)


elif problem_type == "Classification":
    st.subheader("Select the model")
    model = st.selectbox("Select the model", ["Logistic Regression", "Random Forest", "Decision Tree", 
    "Support Vector Machine", "K-Nearest Neighbors", "Gradient Boosting"])
    
    if model == "Logistic Regression":
            st.subheader("Logistic Regression")
            model = LogisticRegression()
    elif model == "Random Forest":
            st.subheader("Random Forest")
            model = RandomForestClassifier()
    elif model == "Decision Tree":    
            st.subheader("Decision Tree")
            model = DecisionTreeClassifier()
    elif model == "Support Vector Machine":
            st.subheader("Support Vector Machine")
            model = SVC()
    elif model == "K-Nearest Neighbors":    
            st.subheader("K-Nearest Neighbors")    
            model = KNeighborsClassifier()    
    elif model == "Gradient Boosting":
            st.subheader("Gradient Boosting")    
            model = GradientBoostingClassifier()

    # Training the model
    st.subheader("Train the model")
    model.fit(X_train, y_train)
    st.write("Model is trained now")

    # Making predictions
    if st.button("Make predictions"):
        y_pred = model.predict(X_test)
        st.write("Predictions:", y_pred)
        
    # Evaluate the model
    if st.button("Evaluate the model"):
        y_test_pred =model.predict(X_test)
        confusion_matrix = confusion_matrix(y_test, y_test_pred)
        st.write("Confusion Matrix:\n", confusion_matrix)
        accuracy = accuracy_score(y_test, y_test_pred)
        st.write("Accuracy:", accuracy)
        precision = precision_score(y_test, y_test_pred,average='weighted')
        st.write("Precision:", precision)
        report = classification_report(y_test, y_test_pred, output_dict=True)
        st.header("Model Evaluation")
        st.write("Classification Report:")
        st.write(pd.DataFrame(report).T)
    else:
        st.write("Train the model first to make predictions or evaluate.")

    # made the plot for the model
    st.header("Plot the model")
    y_test_pred =model.predict(X_test)
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_test_pred)    
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted Values")
    st.pyplot(fig)

else:
    st.write("Please select either 'Regression' or 'Classification' as the problem type.")

import joblib

# Save the model
if st.button("Save Model"):
    joblib.dump(model, "model.joblib")
    st.write("Model saved successfully.")

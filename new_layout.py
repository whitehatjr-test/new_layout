import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Iris Prediction", layout="wide")

# Loading the dataset.
iris_df = pd.read_csv("https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the dataset into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating an SVC model. 
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
score = svc_model.score(X_train, y_train)

@st.cache()
# Create a function 'prediction()' which accepts SepalLength, SepalWidth, PetalLength, PetalWidth as input and returns species name.
def prediction(SepalLength, SepalWidth, PetalLength, PetalWidth):
  species = svc_model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
  species= int(species)
  if species == 0:
    return "Iris-setosa"
  elif species == 1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor"


st.title("Iris Prediction")

st.markdown("""
    Let us predict the species of the **Iris** flower using `Support Vector Machine`
    """)

with st.beta_expander("Show DataFrame"):
    st.table(iris_df)


st.set_option('deprecation.showPyplotGlobalUse', False)

st.header("EDA")

c1, c2, c3 = st.beta_columns((1,2,1))


if c1.checkbox("Show all column names"):
    c1.table(iris_df.columns)


if c2.checkbox("Show summary"):
    c2.table(iris_df.describe())


species_option = c3.selectbox('View column data',('sepal_length','sepal_width','petal_length','petal_width','species'))
if species_option == 'sepal_length':
    c3.write(iris_df['SepalLengthCm'])
elif species_option == 'sepal_width':
    c3.write(iris_df['SepalWidthCm'])
elif species_option == 'petal_length':
    c3.write(iris_df['PetalLengthCm'])
elif species_option == 'petal_width':
    c3.write(iris_df['PetalWidthCm'])
elif species_option == 'species':
    c3.write(iris_df['Species'])
else:
    c3.write("Select A Column")
    
col1, col2, col3 = st.beta_columns(3)

with col1:
    if st.checkbox("Line Chart"):
        st.subheader("Line Chart")
        st.line_chart(iris_df)

with col2:
    # Show Correlation Plots
    if st.checkbox("Simple Correlation Plot "):
        sns.heatmap(iris_df.corr(),annot=True)
        st.pyplot()

with col3:
    # Show Plots
    if st.checkbox("Count plot"):

        sns.countplot('Species',data=iris_df)
        st.pyplot()
        

st.header("Model prediction")

c1, c2 = st.beta_columns((2,1))
with c1:
    inputs = st.beta_container()
    inputs.subheader("Select Values:")
    # Add 4 sliders and store the value returned by them in 4 separate variables. 
    s_len = inputs.slider("Sepal Length", float(iris_df["SepalLengthCm"].min()), float(iris_df["SepalLengthCm"].max()))
    # The 'float()' function converts the 'numpy.float' values to Python float values.
    s_wid = inputs.slider("Sepal Width", float(iris_df["SepalWidthCm"].min()), float(iris_df["SepalWidthCm"].max()))
    p_len = inputs.slider("Petal Length", float(iris_df["PetalLengthCm"].min()), float(iris_df["PetalLengthCm"].max()))
    p_wid = inputs.slider("Petal Width", float(iris_df["PetalWidthCm"].min()), float(iris_df["PetalWidthCm"].max()))

with c2:
    st.subheader("Prediction results with evaluation metrics:")
    if inputs.button("Predict"):

        species_type = prediction(s_len, s_wid, p_len, p_wid)
        st.write("Species predicted:", species_type)
        st.write("Accuracy score of this model is:", score)
        st.write("we can display confusion matrix, classification report, rmse, and many other")






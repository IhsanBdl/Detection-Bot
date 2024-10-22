from pycaret.classification import *
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title='Milk Classification', layout='wide',
                   #initial_sidebar_state=st.session_state.get('sidebar_state', 'collapsed'),
)

# load model yang telah dilatih untuk klasifikasi
data = pd.read_csv('milk.csv')
s = setup(data, target='Grade', session_id=123)
model = load_model('milk_pipeline')

# fungsi untuk melakukan prediksi
def predict(model, input_df):
    predictions_df = predict_model(model, data=input_df)
    st.write(predictions_df)
    grade = predictions_df['prediction_label'][0]
    return grade

def explain_owner():
    st.title('Yeorobun, hello! ☘')
    st.subheader("Get to know me")
    st.info("""
    I'm Abdul Muffid, diving deep into the world of data as a student majoring in Applied Data Science at the Electronic Engineering Polytechnic Institute of Surabaya. Beyond just hitting the books, I'm genuinely captivated by the endless possibilities data offers across different fields.
    
    You might be wondering about this web platform I've whipped up. Well, it's not just a random project—it's part of my journey in MLOps (that's Machine Learning Operations). This web serves classification models for making predictions. It's pretty cool stuff, and I'm excited to see where it takes me. Feel free to explore around and see what insights we can uncover together!
    """)
    
    st.subheader('Catch me now!')
    socmed_option=option_menu(
        menu_title = None,
        options=["Instagram", "GitHub", "LinkedIn"],
        icons=["instagram", "github", "linkedin"],
        orientation="horizontal",
        styles={"nav-link-selected": {"background-color": "lightblue"},},
    )
    
    if socmed_option:
        if socmed_option == "Instagram":
            st.markdown("[<button style='background-color: transparent; color: #125688; padding: 8px 15px; border: 1px solid #125688; border-radius: 4px; cursor: pointer;'>Visit Instagram</button>](https://www.instagram.com/abd_muffid)", unsafe_allow_html=True)
        elif socmed_option == "GitHub":
            st.markdown("[<button style='background-color: transparent; color: #333333; padding: 8px 15px; border: 1px solid #333333; border-radius: 4px; cursor: pointer;'>Visit GitHub</button>](https://github.com/abdmuffid)", unsafe_allow_html=True)
        elif socmed_option == "LinkedIn":
            st.markdown("[<button style='background-color: transparent; color: #0e76a8; padding: 8px 15px; border: 1px solid #0e76a8; border-radius: 4px; cursor: pointer;'>Visit LinkedIn</button>](https://www.linkedin.com/in/abdul-muffid-9065b3246)", unsafe_allow_html=True)
    
def explain_dataset():
    st.title("Milk Dataset")
    st.subheader("About Dataset")
    st.info("""
    This dataset is manually collected from observations. It helps us to build machine learning models to predict the quality of milk. This dataset consists of 7 independent variables ie pH, Temperature, Taste, Odor, Fat, Turbidity, and Color. Generally, the Grade or Quality of the milk depends on these parameters. These parameters play a vital role in the predictive analysis of the milk. You can download this data from [Kaggle](https://www.kaggle.com/datasets/cpluzshrijayan/milkquality).
            """)
    
    if st.button("Preview Data"):
        st.write(data.head())
    
def display_summary_statistics(data):
    st.subheader("Statistic Descriptive")
    st.write("Statistic Descriptive for numerical features:")
    st.write(data.describe())

    # Tampilkan histogram untuk fitur numerik
    st.write("Histogram for numerical features:")
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

    num_plots = len(numeric_columns)
    num_rows = 2
    num_cols = 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Menambahkan jarak antara subplot

    plot_index = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if plot_index < num_plots:
                col = numeric_columns[plot_index]
                axes[i, j].hist(data[col], bins=20, color='skyblue', edgecolor='black')
                axes[i, j].set_title(f'Histogram {col}')
                axes[i, j].set_xlabel(col)
                axes[i, j].set_ylabel('Frequency')
                plot_index += 1
            else:
                axes[i, j].axis('off')

    st.pyplot(fig)

    # Tampilkan count plot untuk fitur kategori
    st.write("Count Plot for categorical features:")
    categorical_columns = data.select_dtypes(include=['object']).columns

    num_plots = len(categorical_columns)
    num_rows = 1  # karena kita hanya ingin 1 baris untuk plot kategori
    num_cols = 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4))

    plot_index = 0
    for j in range(num_cols):
        if plot_index < num_plots:
            col = categorical_columns[plot_index]
            value_counts = data[col].value_counts()
            axes[j].bar(value_counts.index, value_counts.values, color='skyblue', edgecolor='black')
            axes[j].set_title(f'Count Plot {col}')
            axes[j].set_xlabel(col)
            axes[j].set_ylabel('Count')
            axes[j].tick_params(axis='x', rotation=45)  # Menambahkan rotasi untuk sumbu x
            plot_index += 1
        else:
            axes[j].axis('off')  # Tidak menampilkan sumbu jika sudah mencapai jumlah plot maksimum

    st.pyplot(fig)

def display_features_importance(model):
    feature_importance = model.feature_importances_
    features = data.columns[:-1]
    sorted_indices = feature_importance.argsort()[::-1]
    sorted_features = [features[i] for i in sorted_indices]
    sorted_importance = feature_importance[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importance, color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    st.pyplot(plt)

# Fungsi untuk menjelaskan model
def explain_model():
    st.title("Random Forest Classifier")
    st.info("Random Forest is a popular and powerful ensemble learning method used for classification tasks in machine learning. It operates by constructing a multitude of decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees. Each decision tree in the Random Forest is trained on a random subset of the training data and a random subset of features, which helps to reduce overfitting and increase the model's generalization ability. During prediction, each tree in the forest independently predicts the class, and the final prediction is determined by aggregating the votes or predictions of all the trees. Random Forest is known for its high accuracy, like this model that have accuracy around 99%.")

    # Menampilkan setup model
    st.subheader("Setup Model")
    setups = pull()
    st.write(setups)
    
    # Menampilkan fitur penting
    st.subheader("Features Importance")
    display_features_importance(model)

def predict_grade():
    st.title("Let's Predict the Grade of your Milk!!")
    st.subheader('Select one below:')
    data_input_option = option_menu(
        menu_title=None,
        options=["CSV File", "Input Data"],
        icons=["filetype-csv", "123"],
        default_index=0,
        orientation="horizontal",
        styles={"nav-link-selected": {"background-color": "lightblue"},},
        )

    if data_input_option == "CSV File":
        # Input data by uploading CSV file
        uploaded_file = st.file_uploader("Upload CSV file (Note: CSV file must contain all variables as same as the model.)", type=['csv'])
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.write(input_df)  

                if st.button("Predict"):
                    # Check if expected columns are in the DataFrame
                    expected_columns = ['pH', 'Temperature', 'Taste', 'Odor', 'Fat', 'Colour']
                    if all(col in input_df.columns for col in expected_columns):
                        output = predict(model=model, input_df=input_df)
                        st.snow()
                    else:
                        st.image('Images/Error CSV.svg', width=700)
            except pd.errors.ParserError:
                st.error("The uploaded file is not a valid CSV file.")

    else:
        # Input data secara manual
        pH = st.number_input('pH', min_value=0.0, max_value=14.0, value=6.6, step=0.1)
        temperature = st.number_input('Temperature (°C)', min_value=-20, max_value=100, value=35)
        
        col1, col2 = st.columns(2)
        with col1:
            taste = st.selectbox('Taste', ['0', '1'])
            odor = st.selectbox('Odor', ['0', '1'])
        with col2:
            fat = st.selectbox('Fat', ['0', '1'])
            turbidity = st.selectbox('Turbidity', ['0', '1'])
            
        colour = st.number_input('Colour', min_value=0, max_value=255, value=254)

        # Tombol prediksi
        if st.button("Predict"):
            input_dict = {'pH': pH,
                          'Temperature': temperature,
                          'Taste': int(taste),
                          'Odor': int(odor),
                          'Fat': int(fat),
                          'Turbidity': turbidity,
                          'Colour': colour}
            input_df = pd.DataFrame([input_dict])
            output = predict(model=model, input_df=input_df)

            # Menampilkan gambar
            if output == 'high':
                st.image('Images/High.svg', width=700)
                st.snow()
            elif output == 'medium':
                st.image('Images/Medium.svg', width=700)
                st.snow()
            elif output == 'low':
                st.image('Images/Low.svg', width=700)
                st.snow()
            else:
                st.warning('Grade not recognized')
                
def run():
    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=["Introduction", "Information", "Prediction"],
            icons=["person-fill", "clipboard-data", "stars"],
            #menu_icon="cast",
            default_index=0,
            styles={"nav-link-selected": {"background-color": "lightblue"},},
        )
    
    st.sidebar.warning("Yo, check out this dope web app for predicting milk grades! It's lit and super easy to use, fam.")
    st.sidebar.success("Abdul Muffid - 3322600021 - SDT 2022")
    st.sidebar.image('Images/Logo PENS SDT.svg', width=300)
    
    if selected == "Introduction":
        explain_owner()
    elif selected == "Information":
        tab_data, tab_model = st.tabs(["Dataset Info", "Explain Model"])
        with tab_data:
            explain_dataset()
            display_summary_statistics(data)
        with tab_model:
            explain_model()
    elif selected == "Prediction":
        predict_grade()

if __name__ == '__main__':
    run()
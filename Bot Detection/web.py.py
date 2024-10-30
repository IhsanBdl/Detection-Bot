from pycaret.classification import *
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import webbrowser
import matplotlib.pyplot as plt
from datetime import datetime
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

st.set_page_config(page_title='User Profile Classification', layout='wide')

# Meangambil model pickle
model = load_model('wasm_pipeline')

# fungsi untuk melakukan prediksi
def predict(model, input_df):
    predictions_df = predict_model(model, data=input_df)
    st.write(predictions_df)
    profile_class = predictions_df['prediction_label'][0]
    return profile_class

def explain_owner():
    st.title('Yeorobun, hello! ☘')
    st.subheader("Introduction Us! ✨")

    # Intro dengan font yang lebih menarik
    st.markdown("""
    <div style='text-align: left; font-size: 20px;'>
        Hey there! 👋 We’re Abdul Muffid and Maulana Ihsan Athallah, two Applied Data Science students from the Electronic Engineering Polytechnic Institute of Surabaya! 🎓✨
        <br><br>
        Right now, we’re working on a super fun project: an Automated Detection Bot on platform X! 🤖💥 It’s not just any project—it’s our little playground where we get to tackle challenges and uncover hidden gems. 💎
        So come along for the ride, and let’s discover some awesome insights together! 🎉🚀
    </div>
    """, unsafe_allow_html=True)

    st.subheader('Catch Us Now!')

    # Menu Profil
    profile_option = option_menu(
        menu_title=None,
        options=["Profiles"],
        icons=["person"],
        orientation="horizontal",
        styles={
            "nav-link": {"--hover-color": "#EEE"},
            "nav-link-selected": {"background-color": "#FFA500"},
        },
    )

    # Menampilkan kedua profil
    if profile_option == "Profiles":
        # Membuat dua kolom
        col1, col2 = st.columns(2)

        # Profil Ihsan
        with col1:
            st.markdown("""
            <div style='text-align: center;'>
                <img src='https://pbs.twimg.com/profile_images/1712244129898668032/7Q0mP-Xz_400x400.jpg' style='width: 200px; border-radius: 50%;'>
                <h3>Maulana Ihsan Athallah</h3>
                <p>NRP: 123456789</p>
            </div>
            """, unsafe_allow_html=True)

            # Tombol Streamlit dengan hover effect
            if st.button("Visit Instagram (Ihsan)"):
                webbrowser.open_new_tab("https://www.instagram.com/ihsan_profile_link")

        # Profil Muffid
        with col2:
            st.markdown("""
            <div style='text-align: center;'>
                <img src='https://pbs.twimg.com/profile_images/1587293695602470913/UCzJC887_400x400.jpg' style='width: 200px; border-radius: 50%;'>
                <h3>Abdul Muffid</h3>
                <p>NRP: 987654321</p>
            </div>
            """, unsafe_allow_html=True)

            # Tombol Streamlit dengan hover effect
            if st.button("Visit Instagram (Muffid)"):
                webbrowser.open_new_tab("https://www.instagram.com/Muffid_profile_link")

    # Menambahkan pengaturan CSS untuk lebih memperindah tampilan
    st.markdown("""
        <style>
            .stButton > button {
                display: block;
                margin: 0 auto;
                background-color: #FFA500;
                color: white;
                border-radius: 10px;
                padding: 10px;
            }
            .stButton > button:hover {
                background-color: #FF8500;
                color: white;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            }
            img {
                border: 2px solid #FFA500;
                padding: 5px;
                margin-bottom: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

def predict_profile():
    st.markdown("<h2 style='color: #FFA500;'>Automated Detection Bot Accounts on platform X! 🤖💥</h2>", unsafe_allow_html=True)
    st.subheader('Input The Profile Accounts Details Below!')

    # Menggunakan expander untuk semua input data
    with st.expander("Input Profile Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input('Name')
            username = st.text_input('Username')
            profile_image = st.text_input('Profile Image URL')
            account_created_str = st.text_input('Account Created (MM-YYYY)', value='01-2020')
        
        with col2:
            bio = st.text_input('Bio')
            tweet_count = st.number_input('Tweet Count', min_value=0, value=0)
            following = st.number_input('Following Count', min_value=0, value=0)
            followers = st.number_input('Followers Count', min_value=0, value=0)
    
        # Parsing input bulan dan tahun menjadi objek datetime
        try:
            account_created_date = datetime.strptime(account_created_str, '%m-%Y')
            current_date = datetime.now()
            
            # Menghitung umur akun dalam bulan
            age_account = (current_date.year - account_created_date.year) * 12 + (current_date.month - account_created_date.month)
        except ValueError:
            st.error("Format tanggal salah. Harap masukkan dalam format MM-YYYY.")
            return

    # Tombol prediksi dengan spinner dan delay 5 detik
    if st.button("🔍 Predict"):
        input_dict = {'Name': name,
                      'Username': username,
                      'Bio': bio,
                      'default_profile_image': profile_image,
                      'Account Age (Months)': age_account,
                      'Tweet Count': tweet_count,
                      'Following Count': following,
                      'Follower Count': followers}
        input_df = pd.DataFrame([input_dict])
        
        # Simulasi prediksi dengan delay dan spinner
        with st.spinner('Predicting... Please wait for 5 seconds'):
            time.sleep(5)  # Menunggu selama 5 detik
        
        output = predict(model=model, input_df=input_df)
        st.success(f'Profile classification: {output}')
        st.snow()

def run():
    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=["Introduction", "Prediction"],
            icons=["person-fill", "clipboard-data", "stars"],
            default_index=0,
            styles={"nav-link-selected": {"background-color": "lightblue"}},
        )

    st.sidebar.success("Yo, check out this dope web app for predicting user profile classes!")

    if selected == "Introduction":
        explain_owner()
    elif selected == "Prediction":
        predict_profile()

if __name__ == '__main__':
    run()
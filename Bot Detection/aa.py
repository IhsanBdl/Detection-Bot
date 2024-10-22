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
                <img src='https://pbs.twimg.com/profile_images/1843923993168093184/Mvum703E_400x400.jpg' style='width: 200px; border-radius: 50%;'>
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

# Fungsi untuk scraping data profil dari Twitter
def scrape_profile_data(profile_url):
    # Menggunakan webdriver Chrome
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    
    try:
        # Buka halaman profil
        driver.get(profile_url)
        time.sleep(3)  # Tunggu halaman dimuat

        # Ambil nama pengguna
        name = driver.find_element(By.XPATH, "//span[@dir='auto']").text
        
        # Ambil username
        username = driver.find_element(By.XPATH, "//div[@data-testid='UserName']").text
        
        # Ambil bio
        bio = driver.find_element(By.XPATH, "//div[@data-testid='UserDescription']//span").text
        
        # Ambil jumlah tweet
        tweet_count = driver.find_element(By.XPATH, "//a[contains(@href, '/with_replies')]//span[1]").text
        
        # Ambil following count
        following = driver.find_element(By.XPATH, "//a[contains(@href, '/following')]//span[1]").text
        
        # Ambil follower count
        followers = driver.find_element(By.XPATH, "//a[contains(@href, '/followers')]//span[1]").text
        
        # Ambil tanggal akun dibuat
        account_creation_date_str = driver.find_element(By.XPATH, "//div[@data-testid='UserProfileHeader_Items']//span").text
        account_creation_date = datetime.strptime(account_creation_date_str, '%B %Y')  # Ubah format sesuai kebutuhan
        current_date = datetime.now()
        account_age_months = (current_date.year - account_creation_date.year) * 12 + (current_date.month - account_creation_date.month)

        # Kembalikan data yang diambil
        profile_data = {
            'name': name,
            'username': username,
            'bio': bio,
            'tweet_count': int(tweet_count.replace(',', '')),
            'following': int(following.replace(',', '')),
            'followers': int(followers.replace(',', '')),
            'account_age': account_age_months
        }
        return profile_data
    except Exception as e:
        st.error(f"Error: {e}")
        return None
    finally:
        driver.quit()

# Fungsi utama di Streamlit
def predict_profile():
    st.title("Automated Detection Bot Accounts on platform X! 🤖💥")
    
    # Input URL profil
    profile_url = st.text_input("Enter the Profile URL (e.g., Twitter/X URL):")
    
    if st.button("Predict"):
        if profile_url:
            # Scraping data dari URL profil
            profile_data = scrape_profile_data(profile_url)
            
            if profile_data:
                # Mengambil data yang relevan untuk prediksi
                name = profile_data['name']
                username = profile_data['username']
                bio = profile_data['bio']
                tweet_count = profile_data['tweet_count']
                following = profile_data['following']
                followers = profile_data['followers']
                account_age = profile_data['account_age']
                
                # Tampilkan data yang di-scrape ke pengguna
                st.write(f"Name: {name}")
                st.write(f"Username: {username}")
                st.write(f"Bio: {bio}")
                st.write(f"Tweet Count: {tweet_count}")
                st.write(f"Following Count: {following}")
                st.write(f"Followers Count: {followers}")
                st.write(f"Account Age: {account_age} months")
                
                # Lakukan prediksi (contoh sederhana, ganti dengan model ML)
                st.success(f'Predicted Profile Classification: Bot/Not Bot')
            else:
                st.error("Failed to retrieve profile data. Please check the URL or try again.")
        else:
            st.warning("Please enter a valid profile URL.")

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

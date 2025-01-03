import pickle
from pathlib import Path

import streamlit_authenticator as stauth

names = ["Piyush Chittauria", "Shivam Gupta"]
usernames = ["piyushp", "shivam12"]
passwords = ["abc123", "123abc"]

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)
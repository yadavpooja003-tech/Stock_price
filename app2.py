import streamlit as st
import pandas as pd
st.title("Streamlit app:tea:")
st.subheader("Brewed with streamlit")
st.text("Welcome to your first interactive app")
st.write("Choose your fav. variety of tea")
tea = st.selectbox("Your favriote chai:",["Masala tea","Rose tea","Black tea","Green tea","Kesar tea"])
st.write(f"You choose {tea}.Excellent choice")
st.success("Your tea has been brewed")

st.write("pick one programming language")
pro_lan=st.selectbox("Programming language are:",["JAVA","PYTHON","JAVA SCRIPT","PHP"])
st.write(f"You choose {pro_lan}.Excellent choice")
st.success("Now Start coding in new Language")

if st.button("Make chai"):
    st.success("Your chai is being brewed")


add_masala=st.checkbox("Add Masala")   
if add_masala:
    st.write("Masala added to your chai") 

tea_type = st.radio("Pick your chai base:",["Milk","Water","Sugar"])
st.write(f"Selected base {tea_type}.")
flavour = st.selectbox("Choose flavour:",["Adrak","Kesar","Tulsi","long"])
st.write(f"Selected Flavour {flavour}")

sugar = st.slider("sugar value(spoon):",0,6,3)
st.write(f"seleced add sugar level is: {sugar}  spoon")
st.number_input("how many cups",min_value=1,max_value=10,step=1)
name = st.text_input("Enter your name")
if name :
    st.write(f"Welcome , {name} ! Your chai is on the way ")
from datetime import date
dob = st.date_input("Select yoyr date of birth")    
# st.write(f"your date of birth {dob}")
today = date.today()
age = today.year - dob.year-((today.month,today.day)<(dob.month,dob.day))
st.write(f"your age is: {age} years")
uploaded_file = st.file_uploader("Upload csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df) 


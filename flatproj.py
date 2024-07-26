import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import sklearn
import re
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV



# --------------------------------------------------Logo & details on top

icon = Image.open("cpr.png")
st.set_page_config(page_title= " Singapore  Resale Flat Prices Predicting | By Dhanalakshmi S",
                   page_icon= icon,
                   layout= "wide",
                   initial_sidebar_state= "expanded")
st.markdown(f""" <style>.stApp {{
                    background: url('https://mcdn.wallpapersafari.com/medium/68/73/9iD0Nn.jpg');   
                    background-size: cover}}
                 </style>""",unsafe_allow_html=True)
with st.sidebar:
    opt = option_menu("Menu",
                    ['HOME',"PREDICT SELLING PRICE"])
if opt=="HOME":
        st.title(''':rainbow[_Singapore  Resale Flat Prices Predicting_]''')
        st.write("#")
        
        st.write(" This project aims to construct a machine learning model and implement "
                "it as a user-friendly online application in order to provide accurate predictions about the "
                "resale values of apartments in Singapore.")
        st.write(" This prediction model will be based on past transactions "
                "involving resale flats, and its goal is to aid both future buyers and sellers in evaluating the "
                "worth of a flat after it has been previously resold. ")
        st.write("Resale prices are influenced by a wide variety "
                "of criteria, including location, the kind of apartment, the total square footage, and the length "
                "of the lease. The provision of customers with an expected resale price based on these criteria is "
                "one of the ways in which a predictive model may assist in the overcoming of these obstacles.")
        st.write(" ")
        st.markdown("### :orange[DOMAIN :] Real Estate")  
        st.markdown("""
                     ### :orange[TECHNOLOGIES USED :] 

                        - PYTHON
                        - PANDAS
                        - NUMPY
                        - DATA PREPROCESING
                        - EDA
                        - SCIKIT - LEARN
                        - STREAMLIT
                       
                    """)


if opt == "PREDICT SELLING PRICE":
        # Define unique values for select boxes
        flat_model_options = ['IMPROVED', 'NEW GENERATION', 'MODEL A', 'STANDARD', 'SIMPLIFIED',
                              'MODEL A-MAISONETTE', 'APARTMENT', 'MAISONETTE', 'TERRACE', '2-ROOM',
                              'IMPROVED-MAISONETTE', 'MULTI GENERATION', 'PREMIUM APARTMENT',
                              'ADJOINED FLAT', 'PREMIUM MAISONETTE', 'MODEL A2', 'DBSS', 'TYPE S1',
                              'TYPE S2', 'PREMIUM APARTMENT LOFT', '3GEN']
        flat_type_options = ['1 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE', 'MULTI GENERATION']
        town_options = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT TIMAH',
                        'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG',
                        'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE',
                        'QUEENSTOWN', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS',
                        'YISHUN', 'LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS', 'PUNGGOL']
        storey_range_options = ['10 TO 12', '04 TO 06', '07 TO 09', '01 TO 03', '13 TO 15', '19 TO 21',
                                '16 TO 18', '25 TO 27', '22 TO 24', '28 TO 30', '31 TO 33', '40 TO 42',
                                '37 TO 39', '34 TO 36', '46 TO 48', '43 TO 45', '49 TO 51']

        # Streamlit app title
        #st.title("Resale Price Prediction App")

        # Title for the flat details section
       # st.title("Flat Details")
        st.write( f'<h1 style="color:pink;">Flat Details</h1>', unsafe_allow_html=True )
       # st.write('<h1 style="color:pink;"><i>Flat Details</i></h1>', unsafe_allow_html=True)


        col1, col2 = st.columns([5, 5])
        with col1:
            # Main page input fields
            town = st.selectbox("Town", options=town_options)
            flat_type = st.selectbox("Flat Type", options=flat_type_options)
            flat_model = st.selectbox("Flat Model", options=flat_model_options)
        with col2:
            storey_range = st.selectbox("Storey Range", options=storey_range_options)
            floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=0.0, max_value=500.0, value=100.0)
            current_remaining_lease = st.number_input("Current Remaining Lease", min_value=0.0, max_value=99.0, value=20.0)

            # Additional calculations
            year = 2024
            lease_commence_date = current_remaining_lease + year - 99
            years_holding = 99 - current_remaining_lease

        submit_button = st.button("Predict Resale Price")

        # Create a button to trigger the prediction
        if submit_button:
            # Prepare input data for prediction
            input_data = pd.DataFrame({
                'town': [town],
                'flat_type': [flat_type],
                'flat_model': [flat_model],
                'storey_range': [storey_range],
                'floor_area_sqm': [floor_area_sqm],
                'current_remaining_lease': [current_remaining_lease],
                'lease_commence_date': [lease_commence_date],
                'years_holding': [years_holding],
                'remaining_lease': [current_remaining_lease],
                'year': [year]
            })

            # Validation check
            flag = 0
            pattern = "^(?:\d+|\d*\.\d+)$"
            for i in [floor_area_sqm, current_remaining_lease]:
                if not re.match(pattern, str(i)):
                    flag = 1
                    invalid_value = i
                    break

            if flag == 1:
                st.write(f"You have entered an invalid value: {invalid_value}")
            else:
                with open(r'dtmodel.pkl', 'rb') as file:
                    best_model = pickle.load(file)
                with open(r'dtscaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)
                with open(r'dtitype.pkl', 'rb') as f:
                    it = pickle.load(f)
                with open(r'dtstatus.pkl', 'rb') as f:
                    s = pickle.load(f)
                with open(r'dtflat.pkl', 'rb') as f:
                    ft = pickle.load(f)
                with open(r'dtmod.pkl', 'rb') as f:
                    m = pickle.load(f)

                # Prepare the sample for prediction
                sample = np.array([[town, flat_type, flat_model, storey_range, np.log(float(floor_area_sqm)), np.log(float(current_remaining_lease))]])
                sample_it = it.transform(sample[:, [0]]).toarray()
                sample_ft = ft.transform(sample[:, [1]]).toarray()
                sample_m = m.transform(sample[:, [2]]).toarray()
                sample_s = s.transform(sample[:, [3]]).toarray()
                sample = np.concatenate((sample[:, [4, 5]], sample_it, sample_ft, sample_m, sample_s), axis=1)
                sample1 = scaler.transform(sample)
                pred = best_model.predict(sample1)
                predicted_price = np.exp(pred)
                rounded_price = round(predicted_price[0], 2)
        
                st.write('## :violet[Predicted selling price:] ', rounded_price)
      
























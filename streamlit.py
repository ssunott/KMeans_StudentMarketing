import pandas as pd
import pickle
import streamlit as st

# Set the page title and description
st.title("Student Marketing Clustering")
st.write("""
This app groups high school students into two clusters based on their social media profiles.""")

# # Optional password protection (remove if not needed)
# password_guess = st.text_input("Please enter your password?")
# # this password is stores in streamlit secrets
# if password_guess != st.secrets["password"]:
#     st.stop()

# Load dataset to get dropdown values
df = pd.read_csv("data/processed/clustered_data.csv")

# Load the pre-trained model and encoder
with open("models/kmeans_model.pkl", "rb") as f:
    kmean_model = pickle.load(f)


# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("Student Profile")
    
    gradyear = st.selectbox("Graduation Year", sorted(df["gradyear"].unique()))
    gender = st.radio("Gender", ["M", "F", "Unknown"])
    age = st.number_input("Age", min_value=15, max_value=20)
    num_friends = st.number_input("Number of Friends", min_value=0, max_value=500, value=10)

    st.markdown("**Select interests (check all that apply):**")
    # list of all the "key word" columns (sports/interests)
    keyword = [
        "basketball", "football", "soccer", "softball", "volleyball",
        "swimming", "cheerleading", "baseball", "tennis", "sports",
        "cute", "sex", "sexy", "hot", "kissed", "dance", "band",
        "marching", "music", "rock", "god", "church", "jesus", "bible",
        "hair", "dress", "blonde", "mall", "shopping", "clothes",
        "hollister", "abercrombie", "die", "death", "drunk", "drugs"
    ]
    interest = {col: st.checkbox(col.replace("_", " ").title()) for col in keyword}

    # Submit button
    submitted = st.form_submit_button("Submit")


# Handle the dummy variables to pass to the model
if submitted:
    if age is None:
        median_age_per_year = df.groupby('gradyear')['age'].median()
        age = df['gradyear'].map(median_age_per_year)
   
    input_dict = {
        "gradyear": gradyear,
        "age": age,
        "NumberOffriends": num_friends,
        **{col: int(val) for col, val in interest.items()},
        "gender_F": 1 if gender == "F" else 0,
        "gender_M": 1 if gender == "M" else 0,
        "gender_unknown": 1 if (gender != "M" and gender != "F") else 0
    }
    input = pd.DataFrame([input_dict])
    input.to_csv('data/processed/Processed_Input.csv', index=None)
   
    try:
        cluster = kmean_model.predict(input)[0]
        
        st.subheader("Prediction")
        st.write(f"Cluster: **{cluster}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.write(
    """We used the K-Means model to calculate the optimal number of clusters. 
        The Elbow Score and Sihouette Scores of the model are illustrated below."""
)
st.image("sscore.png")
st.image("wcss.png")

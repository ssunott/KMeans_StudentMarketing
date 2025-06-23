import pandas as pd
import pickle
import streamlit as st

# Set the page title and description
st.title("Student Marketing Clustering")
st.write("""
This app groups high school students into clusters based on their social media profiles.""")

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

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("Student Profile")
    
    gradyear = st.selectbox("Graduation Year", sorted(df["gradyear"].unique()))
    gender = st.radio("Gender", ["Male", "Female", "Unknown / Prefer not to say"])
    age = st.number_input("Age", min_value=13, max_value=22)
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
    try:
        if age is None:
            median_age_per_year = df.groupby('gradyear')['age'].median()
            age = df['gradyear'].map(median_age_per_year)
        
        # set gender code
        if gender == "Male":
            gender_code = 1
        elif gender == "Female":
            gender_code = 2
        else:
            gender_code = 3
        
        input_dict = {
            "gradyear": gradyear,
            "gender": gender_code,
            "age": age,
            "NumberOffriends": num_friends,
            **{col: int(val) for col, val in interest.items()},
        }
        
        input = pd.DataFrame([input_dict])
        
        unscaled_cols = ['gradyear', 'gender', 'age', 'NumberOffriends']
        scaled_cols = input.columns[4:40].tolist()
        
        X_input = input[scaled_cols].values
        X_scaled_arr = scaler.transform(X_input)  
        
        df_scaled = pd.DataFrame(X_scaled_arr, columns=scaled_cols)
        df_full = pd.concat([input[unscaled_cols].reset_index(drop=True), df_scaled], axis=1)
        
        df_full.to_csv('data/processed/Processed_Input.csv', index=None)
        
        # Predict the cluster
        cluster = kmean_model.predict(df_full)[0]
        
        st.subheader(f"You belong to Cluster: **{cluster}**!")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
    else:
        st.write("""Below are some fun facts about the clusters:""")
        st.markdown("""
           - **Over 50 %** of all students land in one giant cluster, while the smallest cluster accounts for fewer than 5 % of the crowd.  
            - The cluster with the **youngest** students also boasts the **highest** average friend count—while the **oldest** group has the **fewest**.  
            - No matter the segment, **“music”** reigns supreme—topping the keyword charts in every cluster.  
            - **“God”** and **“dance”** cut across almost all clusters, popping into the top 5 everywhere.  
            - **“Church”** and **“mall”** only make the list for our **smallest** clusters—unique hangouts for those niche groups.  
            - Keywords like **“music,” “dance,”** and **“god”** show the **strongest correlations** with specific clusters, spotlighting the key interests that bind each community.  
            """)
        st.image("cluster_analysis.png", caption="Cluster Analysis")
        st.image("cluster_analysis_topkeywords.png", caption="Cluster Analysis - Top Keywords")
        
        st.write(
            """We used the K-Means model to calculate the optimal number of clusters. 
                The Elbow Score is illustrated below."""
        )
        st.image("wcss.png", caption="Elbow Method for Optimal Clusters")
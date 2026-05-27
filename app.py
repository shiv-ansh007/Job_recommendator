import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from collections import Counter

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="JobPulse AI Dashboard",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("cleaned_jobs.csv")
from sklearn.preprocessing import LabelEncoder

# Encode location

location_encoder = LabelEncoder()

df['location_encoded'] = location_encoder.fit_transform(
    df['location']
)

# Encode experience

experience_encoder = LabelEncoder()

df['experience_encoded'] = experience_encoder.fit_transform(
    df['experience_level']
)
# =====================================================
# LOAD MODEL
# =====================================================

model = pickle.load(
    open("salary_model.pkl", "rb")
)

# =====================================================
# TITLE SECTION
# =====================================================

st.title(" JobPulse AI Dashboard")

st.markdown(
    """
    ### AI-Powered Job Market Analytics & Salary Prediction System
    """
)

st.markdown("---")

# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.header("🔍 Filter Options")

# LOCATION FILTER

selected_locations = st.sidebar.multiselect(
    "Select Locations",
    options=df['location'].unique(),
    default=df['location'].unique()[:5]
)

# EXPERIENCE FILTER

selected_experience = st.sidebar.multiselect(
    "Select Experience Level",
    options=df['experience_level'].unique(),
    default=df['experience_level'].unique()
)

# REMOTE FILTER

selected_remote = st.sidebar.multiselect(
    "Remote Job",
    options=[0, 1],
    default=[0, 1]
)

# SALARY BAND FILTER

selected_salary_band = st.sidebar.multiselect(
    "Salary Band",
    options=df['salary_band'].unique(),
    default=df['salary_band'].unique()
)

# =====================================================
# APPLY FILTERS
# =====================================================

filtered_df = df.copy()

filtered_df = filtered_df[
    filtered_df['location'].isin(selected_locations)
]

filtered_df = filtered_df[
    filtered_df['experience_level']
    .isin(selected_experience)
]

filtered_df = filtered_df[
    filtered_df['remote']
    .isin(selected_remote)
]

filtered_df = filtered_df[
    filtered_df['salary_band']
    .isin(selected_salary_band)
]

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3 = st.tabs([
    "📈 Dashboard",
    "🤖 AI Prediction",
    "🗂 Data Explorer"
])

# =====================================================
# TAB 1 — DASHBOARD
# =====================================================

with tab1:

    # =================================================
    # KPIs
    # =================================================

    st.subheader(" Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Jobs",
        len(filtered_df)
    )

    col2.metric(
        "Average Salary",
        f"${int(filtered_df['salary'].mean())}",
        "+12%"
    )

    col3.metric(
        "Remote Jobs",
        int(filtered_df['remote'].sum())
    )

    col4.metric(
        "Average Skill Count",
        round(filtered_df['skill_count'].mean(), 2)
    )

    st.markdown("---")

    # =================================================
    # TOP HIRING LOCATIONS
    # =================================================

    st.subheader(" Top Hiring Locations")

    location_chart = px.bar(
        filtered_df['location']
        .value_counts()
        .head(10),
        title="Top Hiring Locations"
    )

    st.plotly_chart(
        location_chart,
        use_container_width=True
    )

    # =================================================
    # SALARY DISTRIBUTION
    # =================================================

    st.subheader(" Salary Distribution")

    salary_chart = px.histogram(
        filtered_df,
        x='salary',
        nbins=25,
        title="Salary Distribution"
    )

    st.plotly_chart(
        salary_chart,
        use_container_width=True
    )

    # =================================================
    # EXPERIENCE LEVEL DISTRIBUTION
    # =================================================

    st.subheader(" Experience Level Distribution")

    experience_chart = px.pie(
        filtered_df,
        names='experience_level',
        title="Experience Level"
    )

    st.plotly_chart(
        experience_chart,
        use_container_width=True
    )

    # =================================================
    # SKILL COUNT VS SALARY
    # =================================================

    st.subheader(" Skill Count vs Salary")

    scatter_chart = px.scatter(
        filtered_df,
        x='skill_count',
        y='salary',
        color='experience_level',
        hover_data=['company'],
        title="Skill Count vs Salary"
    )

    st.plotly_chart(
        scatter_chart,
        use_container_width=True
    )

    # =================================================
    # REMOTE VS ONSITE
    # =================================================

    st.subheader(" Remote vs Onsite Jobs")

    remote_chart = px.pie(
        filtered_df,
        names='remote',
        title="Remote Job Distribution"
    )

    st.plotly_chart(
        remote_chart,
        use_container_width=True
    )

    # =================================================
    # TOP SKILLS ANALYSIS
    # =================================================

    st.subheader("Top Skills Demand")

    skills = []

    for tag in filtered_df['tags']:

        tag_list = str(tag).split(',')

        for skill in tag_list:
            skills.append(skill.strip())

    skill_counts = Counter(skills)

    skill_df = pd.DataFrame(
        skill_counts.items(),
        columns=['Skill', 'Count']
    )

    skill_df = skill_df.sort_values(
        by='Count',
        ascending=False
    ).head(10)

    skill_chart = px.bar(
        skill_df,
        x='Skill',
        y='Count',
        title='Top 10 Most Demanded Skills'
    )

    st.plotly_chart(
        skill_chart,
        use_container_width=True
    )

# =====================================================
# TAB 2 — AI SALARY PREDICTION
# =====================================================

with tab2:

    st.subheader("AI Salary Prediction")

    st.markdown(
        """
        Predict expected salary using:
        - skill count
        - remote status
        - location
        - experience level
        """
    )

    # ================================================
    # USER INPUTS
    # ================================================

    skill_count = st.slider(
        "Skill Count",
        1,
        10,
        3
    )

    remote = st.selectbox(
        "Remote Job",
        ["No", "Yes"]
    )

    remote_value = 1 if remote == "Yes" else 0

    # LOCATION

    location_list = sorted(
        df['location'].unique()
    )

    selected_location = st.selectbox(
        "Select Location",
        location_list
    )

    # EXPERIENCE

    experience_list = sorted(
        df['experience_level'].unique()
    )

    selected_experience = st.selectbox(
        "Experience Level",
        experience_list
    )

    # DESCRIPTION LENGTH

    description_length = st.slider(
        "Description Length",
        100,
        5000,
        1000
    )

    # ================================================
    # AUTO ENCODING
    # ================================================

    location_encoded = (
        df[
            df['location'] == selected_location
        ]['location_encoded']
        .iloc[0]
    )

    experience_encoded = (
        df[
            df['experience_level']
            == selected_experience
        ]['experience_encoded']
        .iloc[0]
    )

    # ================================================
    # PREDICTION
    # ================================================

    if st.button("Predict Salary"):

        prediction = model.predict([[
            skill_count,
            remote_value,
            location_encoded,
            experience_encoded,
            description_length
        ]])

        st.success(
            f" Predicted Salary: ${int(prediction[0])}"
        )

# =====================================================
# TAB 3 — DATA EXPLORER
# =====================================================

with tab3:

    st.subheader(" Dataset Explorer")

    # SEARCH BAR

    company_search = st.text_input(
        "Search Company Name"
    )

    if company_search:

        search_df = filtered_df[
            filtered_df['company']
            .str.contains(
                company_search,
                case=False,
                na=False
            )
        ]

        st.dataframe(search_df)

    else:

        st.dataframe(filtered_df)

    # =================================================
    # DOWNLOAD BUTTON
    # =================================================

    csv = filtered_df.to_csv(index=False)

    st.download_button(
        label="⬇ Download Filtered Dataset",
        data=csv,
        file_name="filtered_jobs.csv",
        mime="text/csv"
    )

 
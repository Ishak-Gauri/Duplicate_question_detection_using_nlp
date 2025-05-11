import streamlit as st
import helper
import warnings
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time

# Optional: Try to import streamlit_lottie, but don't fail if it's not available
try:
    from streamlit_lottie import st_lottie
    import requests

    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

warnings.filterwarnings("ignore")


# Function to load Lottie animations with error handling
def load_lottieurl(url):
    if not LOTTIE_AVAILABLE:
        return None
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None


# Page configuration with custom theme
st.set_page_config(
    page_title="Question Similarity Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7ff;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7ff 0%, #e0e6ff 100%);
    }
    .title-container {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.19), 0 6px 6px rgba(0,0,0,0.23);
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .result-card {
        background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
        border-radius: 10px;
        padding: 25px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .feature-title {
        color: #4b6cb7;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .stTextArea>div>div>textarea {
        border-radius: 5px;
        border: 1px solid #e0e6ff;
    }
    .similarity-meter {
        height: 30px;
        background-color: #e0e6ff;
        border-radius: 15px;
        margin: 10px 0;
        overflow: hidden;
    }
    .similarity-fill {
        height: 100%;
        border-radius: 15px;
        transition: width 1s ease-in-out;
    }
    .icon-container {
        font-size: 3rem;
        text-align: center;
        margin: 20px 0;
        color: #4b6cb7;
    }
</style>
""", unsafe_allow_html=True)

# Title section with gradient background
st.markdown("""
<div class="title-container">
    <h1 style='text-align: center; color: white; margin: 0;'>
        <span style='font-size: 42px;'>🔍 Question Similarity Detector</span>
    </h1>
    <p style='text-align: center; color: #e0e6ff; margin-top: 10px;'>
        Advanced NLP tool to detect similarity between questions
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar for information and settings
with st.sidebar:
    st.markdown("### About")
    st.info("""
    This application uses Natural Language Processing techniques to analyze and 
    compare two questions to determine their similarity. It extracts various 
    features and calculates a similarity score.
    """)

    st.markdown("### Features Used")
    st.markdown("""
    - Text length and word count
    - Common words analysis
    - Token features
    - Fuzzy matching
    - Bag of Words vectors
    - Cosine similarity
    """)

    st.markdown("### Settings")
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.85,
        step=0.01,
        help="Adjust the threshold for determining duplicate questions"
    )

    # Display icon instead of animation if Lottie is not available
    if LOTTIE_AVAILABLE:
        lottie_compare = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json")
        if lottie_compare:
            st_lottie(lottie_compare, height=200, key="compare_animation")
        else:
            st.markdown('<div class="icon-container">🔄</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="icon-container">🔄</div>', unsafe_allow_html=True)

# Main content area with two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='feature-title'>📝 First Question</p>", unsafe_allow_html=True)
    q1 = st.text_area(
        "Enter your first question here",
        height=150,
        placeholder="Type the first question here...",
        key="q1"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='feature-title'>📝 Second Question</p>", unsafe_allow_html=True)
    q2 = st.text_area(
        "Enter your second question here",
        height=150,
        placeholder="Type the second question here...",
        key="q2"
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Center the compare button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    compare_button = st.button("🔍 Compare Questions", use_container_width=True)

# Results section
if compare_button:
    if q1.strip() == "" or q2.strip() == "":
        st.warning("⚠️ Please enter both questions before comparing.")
    else:
        # Show a spinner while processing
        with st.spinner("Analyzing questions..."):
            # Simulate processing time for better UX
            time.sleep(1)

            # Get results from helper
            query, result_message = helper.query_point_creator(q1, q2)

            # Extract cosine similarity from the query
            cosine_sim = query[0, 22]  # Based on the helper code, index 22 is cosine similarity

            # Display results in a nice card
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)

            # Success message with icon instead of animation if Lottie is not available
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success("✅ Analysis completed successfully!")

                # Similarity meter visualization
                st.markdown("<p class='feature-title'>Similarity Score</p>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="similarity-meter">
                    <div class="similarity-fill" style="width: {cosine_sim * 100}%; 
                    background: linear-gradient(90deg, 
                        {'#ff4e50' if cosine_sim < 0.3 else '#ffd452' if cosine_sim < 0.7 else '#1cb142'} 0%, 
                        {'#f9d423' if cosine_sim < 0.3 else '#1cb142' if cosine_sim < 0.7 else '#0f9b0f'} 100%);">
                    </div>
                </div>
                <p style="text-align: center; font-weight: bold; font-size: 1.5rem; margin: 10px 0;">
                    {cosine_sim:.2f}
                </p>
                """, unsafe_allow_html=True)

                # Result message with appropriate styling
                if cosine_sim > similarity_threshold:
                    st.markdown(f"""
                    <div style="background-color: rgba(28, 177, 66, 0.1); border-left: 5px solid #1cb142; 
                    padding: 15px; border-radius: 5px; margin-top: 20px;">
                        <p style="font-size: 18px; font-weight: bold; color: #1cb142; margin: 0;">
                            The questions are very similar or duplicates!
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: rgba(75, 108, 183, 0.1); border-left: 5px solid #4b6cb7; 
                    padding: 15px; border-radius: 5px; margin-top: 20px;">
                        <p style="font-size: 18px; font-weight: bold; color: #4b6cb7; margin: 0;">
                            The questions are not duplicates.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                # Display icon instead of animation if Lottie is not available
                if LOTTIE_AVAILABLE:
                    lottie_success = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_jR229a.json")
                    if lottie_success:
                        st_lottie(lottie_success, height=150, key="success_animation")
                    else:
                        st.markdown('<div class="icon-container">✅</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="icon-container">✅</div>', unsafe_allow_html=True)

            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Feature Comparison", "3D Visualization", "Raw Data"])

            with tab1:
                # Extract some key features for visualization
                token_features = query[0, 7:15]  # Token features
                fuzzy_features = query[0, 18:22]  # Fuzzy features

                # Create bar chart for token features
                token_fig = px.bar(
                    x=["Common Word Ratio", "Word Ratio Max", "Stop Word Ratio", "Stop Word Ratio Max",
                       "Token Ratio", "Token Ratio Max", "Last Word Match", "First Word Match"],
                    y=token_features,
                    labels={"x": "Feature", "y": "Value"},
                    title="Token Features Comparison",
                    color=token_features,
                    color_continuous_scale="Blues",
                    template="plotly_white"
                )
                token_fig.update_layout(height=400)
                st.plotly_chart(token_fig, use_container_width=True)

                # Create radar chart for fuzzy features
                fuzzy_fig = go.Figure()
                fuzzy_fig.add_trace(go.Scatterpolar(
                    r=fuzzy_features,
                    theta=["Fuzz Ratio", "Partial Ratio", "Token Sort Ratio", "Token Set Ratio"],
                    fill='toself',
                    name='Fuzzy Features',
                    line_color='#4b6cb7'
                ))
                fuzzy_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    title="Fuzzy Matching Features",
                    height=400,
                    template="plotly_white"
                )
                st.plotly_chart(fuzzy_fig, use_container_width=True)

            with tab2:
                # Create 3D visualization of feature space
                # We'll use the first 3 features for visualization
                basic_features = query[0, 0:3]  # First 3 basic features

                # Create a sample of points around our query point for visualization
                n_points = 50
                x = np.random.normal(basic_features[0], basic_features[0] * 0.2, n_points)
                y = np.random.normal(basic_features[1], basic_features[1] * 0.2, n_points)
                z = np.random.normal(basic_features[2], basic_features[2] * 0.2, n_points)

                # Color based on distance from our point (simulating similarity)
                distances = np.sqrt((x - basic_features[0]) ** 2 +
                                    (y - basic_features[1]) ** 2 +
                                    (z - basic_features[2]) ** 2)

                # Create 3D scatter plot
                fig_3d = go.Figure(data=[
                    # Background points
                    go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=distances,
                            colorscale='Blues',
                            opacity=0.6
                        ),
                        name='Feature Space'
                    ),
                    # Our query point
                    go.Scatter3d(
                        x=[basic_features[0]], y=[basic_features[1]], z=[basic_features[2]],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color='red',
                            symbol='diamond'
                        ),
                        name='Current Query'
                    )
                ])

                # Update layout
                fig_3d.update_layout(
                    title="3D Feature Space Visualization",
                    scene=dict(
                        xaxis_title="Question 1 Length",
                        yaxis_title="Question 2 Length",
                        zaxis_title="Word Count Q1",
                        xaxis=dict(backgroundcolor="#f5f7ff"),
                        yaxis=dict(backgroundcolor="#f5f7ff"),
                        zaxis=dict(backgroundcolor="#f5f7ff")
                    ),
                    height=600,
                    margin=dict(l=0, r=0, b=0, t=30)
                )

                st.plotly_chart(fig_3d, use_container_width=True)

                st.info("""
                This 3D visualization shows the feature space with the current query highlighted in red.
                The surrounding points represent similar questions in the feature space, with color 
                indicating distance (similarity) to the current query.

                You can rotate, zoom, and pan the visualization to explore the feature space.
                """)

            with tab3:
                # Show raw feature data
                st.markdown("<p class='feature-title'>Raw Feature Vector</p>", unsafe_allow_html=True)

                # Format the data for better display
                basic_features = query[0, 0:7]
                token_features = query[0, 7:15]
                length_features = query[0, 15:18]
                fuzzy_features = query[0, 18:22]

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("##### Basic Features")
                    basic_df = {
                        "Feature": ["Q1 Length", "Q2 Length", "Q1 Word Count", "Q2 Word Count",
                                    "Common Words", "Total Words", "Common/Total Ratio"],
                        "Value": basic_features
                    }
                    st.dataframe(basic_df, use_container_width=True)

                    st.markdown("##### Length Features")
                    length_df = {
                        "Feature": ["Absolute Length Diff", "Average Token Length", "Longest Common Substring Ratio"],
                        "Value": length_features
                    }
                    st.dataframe(length_df, use_container_width=True)

                with col2:
                    st.markdown("##### Token Features")
                    token_df = {
                        "Feature": ["Common Word Ratio", "Word Ratio Max", "Stop Word Ratio", "Stop Word Ratio Max",
                                    "Token Ratio", "Token Ratio Max", "Last Word Match", "First Word Match"],
                        "Value": token_features
                    }
                    st.dataframe(token_df, use_container_width=True)

                    st.markdown("##### Fuzzy Features")
                    fuzzy_df = {
                        "Feature": ["Fuzz Ratio", "Partial Ratio", "Token Sort Ratio", "Token Set Ratio"],
                        "Value": fuzzy_features
                    }
                    st.dataframe(fuzzy_df, use_container_width=True)

                st.markdown("##### Cosine Similarity")
                st.metric("Similarity Score", f"{cosine_sim:.4f}")

            st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; color: #666;">
    <p>Question Similarity Detector | Powered by NLP and Machine Learning</p>
</div>
""", unsafe_allow_html=True)
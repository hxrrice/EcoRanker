
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Title of the app
st.title("EcoRanker: Big Data-Enhanced Sustainability Evaluation with COBRA")

# Explanation of the system
st.markdown("""
**EcoRanker** is a decision support system designed to evaluate sustainability rankings using **Big Data** and the **COBRA (Comprehensive Distance-Based Ranking)** method. This system leverages multi-criteria decision-making (MCDM) for evaluating alternative sustainability practices or initiatives.

This app allows users to input sustainability data, run the COBRA algorithm for ranking, and visualize the results.
""")

# Sidebar for user input
st.sidebar.header("User Input Parameters")

# Create a form for data input
n = st.sidebar.slider("Number of Alternatives (States, Companies, etc.)", 1, 20, 5)
criteria = st.sidebar.slider("Number of Criteria (Sustainability Dimensions)", 1, 10, 5)

# User input: Enter sustainability scores for different alternatives and criteria
data = {}
for i in range(1, n + 1):
    st.sidebar.subheader(f"Alternative {i}")
    data[f"Alternative {i}"] = []
    for j in range(1, criteria + 1):
        score = st.sidebar.number_input(f"Score for Criterion {j} (Alternative {i})", min_value=0, max_value=100, value=50)
        data[f"Alternative {i}"].append(score)

# Convert input data into a DataFrame
df = pd.DataFrame(data)
df.index = [f"Criterion {i}" for i in range(1, criteria + 1)]

# Display the input data
st.subheader("Input Sustainability Data")
st.write(df)

# COBRA Method - Distance-Based Ranking (simplified)
def cobra_method(df):
    # Calculate the distance from the ideal solution
    ideal_solution = df.max(axis=0)  # Ideal solution: highest value for each criterion
    distance_to_ideal = np.sqrt(((df - ideal_solution) ** 2).sum(axis=0))  # Euclidean distance to the ideal solution

    # Rank alternatives based on the distance to the ideal solution
    rankings = distance_to_ideal.sort_values(ascending=True).index
    return rankings, distance_to_ideal

# Run the COBRA method on the data
rankings, distance_to_ideal = cobra_method(df)

# Display the rankings
st.subheader("COBRA Rankings")
st.write(f"The rankings of alternatives based on the COBRA method are:")
for i, rank in enumerate(rankings):
    st.write(f"**Rank {i + 1}: {rank}**")

# Plot the rankings
fig = px.bar(x=rankings, y=distance_to_ideal[rankings], labels={'x': 'Alternative', 'y': 'Distance to Ideal Solution'}, title="COBRA Ranking Results")
st.plotly_chart(fig)

# Additional functionality (e.g., sensitivity analysis, Big Data visualization) can be added below.

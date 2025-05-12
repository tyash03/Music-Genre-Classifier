import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Music Genre App", layout="wide")

st.title("🎶 Music Genre Explorer & Classifier")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload your music CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if df.empty:
        st.error("The uploaded CSV is empty!")
        st.stop()

    st.success("CSV loaded successfully!")

    # Check if essential columns are present
    if 'music_genre' not in df.columns or 'artist_name' not in df.columns:
        st.error("Required columns ('music_genre' or 'artist_name') are missing in the dataset!")
        st.stop()

    # Handling missing values for numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())  # Fill numeric NaNs with mean

    # Handling missing values for non-numeric columns
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    for col in non_numeric_columns:
        if df[col].dtype == 'object':  # Handle categorical columns
            df[col] = df[col].fillna(df[col].mode()[0])  # Fill NaN with the mode of the column (most frequent value)
        else:  # If it's a non-categorical column (e.g., datetime)
            df[col] = df[col].fillna('Unknown')  # Replace NaN with 'Unknown'

    # Display the Genre Distribution Table
    st.subheader("📊 Music Genre Distribution")
    genre_counts = df['music_genre'].value_counts()

    if genre_counts.empty:
        st.warning("No genre data available for plotting.")
    else:
        fig1, ax1 = plt.subplots()
        sns.barplot(x=genre_counts.index, y=genre_counts.values, ax=ax1)
        plt.xticks(rotation=45)
        ax1.set_ylabel("Number of Songs")
        ax1.set_xlabel("Genre")
        st.pyplot(fig1)

    # Feature Correlation Heatmap
    st.subheader("🔥 Feature Correlation Heatmap")
    numeric_df = df[numeric_columns].dropna(axis=1, how='any')  # Remove columns with NaN values
    if numeric_df.shape[0] > 1 and numeric_df.shape[1] > 1:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("Not enough numeric data to generate a heatmap.")

    # KNN Accuracy Table
    st.subheader("🤖 KNN Accuracy Table (Predicting Genre)")
    if 'music_genre' in df.columns:
        # Prepare features
        le = LabelEncoder()
        y = le.fit_transform(df['music_genre'])

        if numeric_df.shape[0] > 0 and len(y) > 0:
            X_train, X_test, y_train, y_test = train_test_split(numeric_df, y, test_size=0.2, random_state=42)

            accuracy_list = []
            for k in range(1, 11):
                model = KNeighborsClassifier(n_neighbors=k)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                accuracy_list.append((k, round(acc * 100, 2)))  # Ensure accuracy is a float and rounded

            acc_df = pd.DataFrame(accuracy_list, columns=["k", "Accuracy (%)"])
            st.dataframe(acc_df.set_index("k"))
        else:
            st.warning("Not enough data available for training the model.")
    else:
        st.warning("❗ 'music_genre' column not found for classification.")

    # Genre Dropdown and Artist Display
    st.subheader("🎧 Discover Artists by Genre")
    genres = sorted(df['music_genre'].dropna().unique())
    selected_genre = st.selectbox("Select a genre:", genres)

    artists = df[df['music_genre'] == selected_genre]['artist_name'].dropna().unique()
    st.markdown(f"**🎤 Artists in {str(selected_genre)}:**")  # Ensure genre is a string
    for artist in sorted(artists[:50]):  # limit to 50
        st.markdown(f"- **{str(artist)}**")  # Ensure artist is a string



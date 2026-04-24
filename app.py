import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Career Assistant", page_icon="🤖")

st.title("🤖 AI Career Assistant")
st.write("Enter your skills and get the best career suggestion!")

# Input
user_skills = st.text_input("💡 Enter your skills (comma separated)")

# Career database
careers = {
    "Data Scientist": "Python Machine Learning Statistics Data Analysis SQL",
    "Web Developer": "HTML CSS JavaScript React Node",
    "Cyber Security": "Networking Linux Security Ethical Hacking",
    "AI Engineer": "Python Deep Learning NLP TensorFlow",
    "App Developer": "Java Kotlin Android Flutter"
}

def analyze_all(skills):
    vectorizer = CountVectorizer()
    scores = {}

    for career, skillset in careers.items():
        vectors = vectorizer.fit_transform([skills, skillset]).toarray()
        score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
        scores[career] = score

    return scores

# Button
if st.button("🔍 Analyze Career"):
    if user_skills.strip() == "":
        st.warning("Please enter some skills!")
    else:
        scores = analyze_all(user_skills)

        best_career = max(scores, key=scores.get)
        best_score = scores[best_career]

        st.success(f"🎯 Best Career: {best_career}")
        st.write(f"📊 Match Score: {round(best_score*100,2)}%")

        # 📊 BAR CHART
        st.subheader("📊 Career Match Comparison")

        careers_list = list(scores.keys())
        score_values = [v*100 for v in scores.values()]

        fig, ax = plt.subplots()
        ax.bar(careers_list, score_values)
        ax.set_ylabel("Match %")
        ax.set_title("Career Match Scores")

        st.pyplot(fig)

        # 📈 LINE GRAPH
        st.subheader("📈 Score Trend")

        fig2, ax2 = plt.subplots()
        ax2.plot(careers_list, score_values, marker='o')
        ax2.set_ylabel("Match %")
        ax2.set_title("Score Trend Across Careers")

        st.pyplot(fig2)

        # Suggestions
        st.subheader("📌 Suggested Improvements:")
        st.write("- Learn advanced concepts in this field")
        st.write("- Build 2-3 real-world projects")
        st.write("- Practice problem solving")

        st.subheader("🚀 Future Scope:")
        st.write("This system can be improved using real ML models and LinkedIn data.")

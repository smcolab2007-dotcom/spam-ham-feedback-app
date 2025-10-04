import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data
emails = [
    "Congratulations, you won a free iPhone!", "Meeting scheduled tomorrow at 10 AM",
    "Claim your free prize now!!!", "Please review the attached project report",
    "Win cash now, limited offer", "Lunch at noon?", "Exclusive deal: buy one get one free",
    "Minutes of today's meeting attached", "Get a free gift card by completing this survey!",
    "Team outing next Friday, RSVP", "Limited-time offer! Earn $5000 per week",
    "Can you send me the latest sales figures?", "Your account has been compromised, reset now",
    "Reminder: doctor appointment tomorrow at 9 AM", "Win a luxury vacation to the Bahamas!",
    "Please find the invoice attached", "Earn money fast with this simple trick",
    "Family dinner plans for Saturday?", "Get free tickets to the concert of your dreams!",
    "Notes from today's lecture attached",
]
labels = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]  # 1 = Spam, 0 = Ham

# Train model
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=None, analyzer='char', ngram_range=(3,5))),
    ('clf', MultinomialNB())
])
model.fit(emails, labels)

st.title("Email Spam Detection with Feedback and Chart")
st.info("You can enter up to 5 messages in this session.")

# Initialize session state
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []

# Input and prediction
if len(st.session_state.predictions) < 5:
    input_text = st.text_area("Enter email text:")

    if st.button("Predict"):
        prediction = model.predict([input_text])[0]
        st.session_state.predictions.append(prediction)
        st.success(f"Prediction: {'Spam' if prediction == 1 else 'Ham'}")

# Feedback for the last prediction
if len(st.session_state.predictions) > len(st.session_state.feedback) and len(st.session_state.predictions) <= 5:
    feedback_option = st.radio(
        "Was the prediction correct?",
        ("Enjoyed 😄 (Correct)", "Not Enjoyed 😅 (Incorrect)"),
        key=len(st.session_state.feedback)
    )
    if st.button("Submit Feedback", key="fb" + str(len(st.session_state.feedback))):
        st.session_state.feedback.append(feedback_option)
        st.info(f"Feedback recorded: {feedback_option}")

# Maximum reached message
if len(st.session_state.predictions) >= 5:
    st.warning("Maximum of 5 messages reached in this session.")

# Bar chart for feedback
if st.session_state.feedback:
    correct_count = st.session_state.feedback.count("Enjoyed 😄 (Correct)")
    incorrect_count = st.session_state.feedback.count("Not Enjoyed 😅 (Incorrect)")

    st.subheader("Prediction Feedback Summary")
    fig, ax = plt.subplots()
    ax.bar(["Correct", "Incorrect"], [correct_count, incorrect_count], color=["green", "red"])
    for i, v in enumerate([correct_count, incorrect_count]):
        ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
    st.pyplot(fig)

# Confusion matrix for training data
y_pred = model.predict(emails)
cm = confusion_matrix(labels, y_pred)
st.subheader("Confusion Matrix (Training Data)")
fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig2)

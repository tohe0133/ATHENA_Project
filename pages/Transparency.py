import streamlit as st
hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Used Dataset")
st.markdown("In order to train our models we used a dataset containing a collection of 79,000 articles, which are divided into two categories: 'true' and "
            "'misinformation, fake news, or propaganda'.")
st.write(''' 
- 'True' Articles: There are 34,975 articles labeled as 'true'. These articles are sourced from reputable sources like 
Reuters, the New York Times, the Washington Post, and others. The dataset includes articles from mainstream and credible 
news outlets.
- 'Misinformation, Fake News, or Propaganda' Articles: There are 43,642 articles in this category. These articles originate
from various sources, including American right-wing extremist websites (e.g., Redflag Newsdesk, Breitbart, 
Truth Broadcast Network). Some of the 'fake' articles are part of a previously published dataset cited in
a research paper focused on online fake news detection. Disinformation and propaganda cases collected by 
the EUvsDisinfo project, which fact-checks cases originating from pro-Kremlin media and distributed across
the European Union. 
''')
st.markdown("Possible Biases and Limitations:")
st.write('''
- Source Bias: The 'true' articles are from mainstream sources, while the 'misinformation' category includes extremist 
and pro-Kremlin sources. This could introduce a source bias, potentially making the model less effective at detecting 
other types of misinformation.
- Labeling Bias: The accuracy of labels (true vs. misinformation) can be subjective, leading to potential labeling bias. 
Misclassification may occur, affecting model training and performance. to potential labeling bias. Misclassification 
may occur, affecting model training and performance.
- Selection Bias: Overrepresentation or underrepresentation of certain types of misinformation may lead to biases in 
the model's ability to detect specific categories of fake news. 
- Content Bias: The dataset might not capture the full diversity of misinformation tactics and topics, potentially 
limiting the model's ability to detect less common or emerging forms of fake news. 
- Temporal Bias: The dataset may not reflect evolving strategies used by misinformation sources over time, potentially 
missing recent trends.
- Language Bias: If the dataset is primarily in one language, the model may be less effective at detecting misinformation 
in other languages.
- Geopolitical Bias: The dataset's focus on pro-Kremlin disinformation may not adequately represent misinformation 
sources from other regions, introducing a geopolitical bias.
- Data Cleaning: Removal of article text and inclusion of only specific information may affect the richness of features 
available for model training. \n\n
''')

st.title("Vectorizer")
st.markdown( " **Bag of Words** \n\n "
             "Bag of Words turns text into a numerical representation by comparing the text at hand with a dictionary and counting the occurrence of each word of the dictionary in the given text, without considering the order or structure of the sentences.")
st.markdown(" **TF-IDF** \n\n "
"TF-IDF stands for Term Frequency-Inverse Document Frequency. It's a method to represent the importance of words in a text. Imagine you have a library of books, and you want to determine the significance of a word in a specific book. If a word appears frequently in that book (high Term Frequency), it might be important. However, if the same word appears often in many other books in the library (making it common across documents), its uniqueness or significance diminishes. "

"In essence, TF-IDF gives words a weight based on how frequently they appear in one text compared to their presence in a collection of texts. Words that are frequent in a specific text but rare in the entire collection get a higher weight, signaling their significance.")
        

st.title("Models")
st.markdown("**Logistic Regression** \n\n"
            "Logistic regression is a statistical model used for binary classification tasks. It estimates the probability of an input belonging to one of two classes based on input features. In fake news detection, it's valuable due to its simplicity and interpretability. By analyzing text features like TF-IDF vectors, logistic regression can assign probabilities to news articles being fake or real. It's particularly useful when there's a need for transparency and explanation, as it allows users to understand which features influence the model's decision. Logistic regression provides a straightforward, effective way to classify news articles, aiding in the identification of potentially deceptive content. \n\n"
            "*BOW Metrics*: accuracy = 93%, F1 = 93%, precision = 93%, recall = 93% \n\n"
            "*TF-IDF Metrics*: accuracy = 94%, F1 = 94%, precision = 94%, recall = 94%")

st.markdown("**Support Vector Machine** \n\n"
            "A Support Vector Machine (SVM) is a powerful machine learning algorithm for binary classification. It works by finding the optimal hyperplane that best separates two classes in a high-dimensional space. SVMs are effective for fake news detection due to their ability to handle complex, non-linear relationships in text data. They use a kernel trick to transform data into a higher-dimensional space, making it easier to distinguish between real and fake news articles. SVMs are also robust against overfitting and can provide strong generalization performance. This makes them valuable for identifying deceptive content within a noisy and dynamic news landscape, where patterns may not be linear.\n\n"
            "*BOW Metrics*: accuracy = 93%, F1 = 93%, precision = 93%, recall = 93% \n\n"
            "*TF-IDF Metrics*: accuracy = 94%, F1 = 94%, precision = 94%, recall = 94%")

st.markdown("**XGBoost** \n\n"
            "XGBoost stands for eXtreme Gradient Boosting. It's a method that takes a step-by-step approach to solving a problem. Imagine you're trying to solve a difficult puzzle. Instead of attempting to solve it all at once, you start with one piece, then add another, and another, making corrections as you go, until the picture becomes clearer and you get closer to the solution. "

"In the world of data, XGBoost builds one simple model, checks how well it did, then builds another model to improve on the mistakes of the first, and so on. Each new model tries to correct the errors made by the combination of previous models. By continuously learning from mistakes and adding new models, XGBoost eventually creates a strong combined model that can make accurate predictions. "

"In essence, XGBoost builds solutions incrementally, always learning from past mistakes to make better future decisions.\n\n"
            "*BOW Metrics*: accuracy = 93%, F1 = 93%, precision = 93%, recall = 93% \n\n"
            "*TF-IDF Metrics*: accuracy = 93%, F1 = 93%, precision = 93%, recall = 93%")


st.markdown("**Naive Bayes** \n\n"
            "Naive Bayes is a method that calculates the probability or likelihood of an event based on prior knowledge of conditions that might be related to the event. It's like trying to predict if a team will win a game based on previous game outcomes. If historically the team often wins when playing at home, Naive Bayes would consider that information when predicting the result of a new game played at home. "

"However, what makes it 'naive' is that it assumes each condition (like playing at home, weather, recent performance) influences the outcome independently of the other conditions. It's like saying the advantage of playing at home doesn’t change even if the team has been on a losing streak or if it’s raining. "

"In essence, Naive Bayes predicts outcomes based on historical data, while making the simple assumption that all influencing factors act independently."
 "*BOW Metrics*: accuracy = 86%, F1 = 86%, precision = 86%, recall = 86% \n\n"
            "*TF-IDF Metrics*: accuracy = 85%, F1 = 85%, precision = 85%, recall = 85%")

st.markdown("**Random Forest** \n\n"
            "Imagine a decision tree as a flowchart that asks a series of yes-or-no questions to make a decision. For instance, if you wanted to decide what to wear, a decision tree might start by asking, 'Is it raining?' Depending on your answer, the next question could be about the temperature, and so on. By the end of the questions, the tree helps you pick an outfit. "

"Now, a Random Forest is like gathering a whole group of these decision trees - each with its own slightly different set of questions or way of looking at the data. Not all trees will come to the same conclusion because they might prioritize different aspects of the data. "

"When you use a Random Forest to make a decision, you're essentially getting each tree in this group to cast a vote based on its set of questions. The final decision is made by considering the majority vote from all the trees. Because each tree has its own perspective and they all get a say, the combined decision is often more balanced and accurate than relying on just one tree.\n\n"
            "*Metrics*: accuracy = 79%, F1 = 79%, precision = 80%, recall = 79% \n\n"
            "*TF-IDF Metrics*: accuracy = 86%, F1 = 86%, precision = 86%, recall = 86%")

st.markdown("**BERT** \n\n"
            "Language models like BERT are powered by deep learning, a type of artificial intelligence based on neural networks. These neural networks consist of layers upon layers of interconnected nodes, each contributing to the model's understanding of language. "
"BERT reads sentences in a unique way. Instead of reading word by word from left to right, it considers the entire sentence at once. This “bidirectional” approach helps BERT grasp the context and meaning of words more accurately, making it a powerful tool for various language tasks like translation, question-answering, and more. "
"However, the complex interplay of nodes and weights within the network makes it challenging to pinpoint exactly how BERT arrives at a particular understanding or answer, making it a black box.\n\n"
"*Metrics*: accuracy = 99%, F1 = 99%, precision = 98%, recall = 99%")







import streamlit as st
import requests
import pickle
from spacy.lang.en import English
from nltk.corpus import stopwords
import nltk
import yake
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import 	WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import make_scorer, accuracy_score


## download the dictionary for stopwords
nltk.download('stopwords')

## get the set of stopwords
stop_words_set = set(stopwords.words('english'))

## Load English tokenizer from spacy
nlp = English()
tokenizer = nlp.tokenizer ## make instance

# download Punkt Sentence Tokenizer
nltk.download("punkt")
# dowload WordNet which is a lexical database of English
nltk.download("wordnet")

# Initiate a WordNet Lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

nltk.download('vader_lexicon')

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

def optimized_preprocess(texts): # Takes in a list of texts, i.e. the entire corpus
    result = []

    # Tokenize and Lemmatize using spaCy's tokenizer and WordNet's Lemmatizer
    for text in texts:
        tokens = [wordnet_lemmatizer.lemmatize(token.text.lower()) for token in tokenizer(text) if token.text.isalpha() and token.text.lower() not in stop_words_set]
        tokenization = nltk.word_tokenize(text)
        result.append(" ".join(tokens))
    return result

# Retrieve the OpenAI API key from Streamlit secrets
api_key = st.secrets["openai"]["api_key"]

def GPTExplain(text):
    # API endpoint and headers
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Payload for GPT API
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "system", "content": text}],
        "temperature": 0.7
    }

    # Send a request to the GPT API
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        # Extract the chatbot's response content
        chatbot_response = response.json()["choices"][0]["message"]["content"]
        st.markdown(chatbot_response)
    else:
        st.markdown(f"There was an error with code: {str(response.status_code)}")
    return


# Initialize conversation in session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

st.title("Fake News Detection")
st.markdown("The purpose of this website is to detect whether a text contains disinformation or not. "
            "This is done by using [machine learning models](https://www.databricks.com/glossary/machine-learning-models)"
            " which can learn from and make predictions based on data."
            " [Vectorizers](https://medium.com/swlh/understanding-count-vectorizer-5dd71530c1b) are preprocessing"
            " techniques that convert textual data into numerical format, making it suitable for machine learning "
            "models. Together, machine learning models and vectorizers are essential components of text analysis."
            "  \n\nIn general, there is a trade off between better performing models and how easy the models are to"
            " understand. For the sake of being as transparent as possible we give you as the user the option to "
            "choose if you want an easier to understand model, or a better performing one. The best performing model"
            " and vectorizer will be pre-selected for you. You can then paste your full article bellow and press enter."
            "  \n\nFor more information about models and vectorizers, as well as the specific ones used on this website, "
            "you can hover the question marks by the selection boxes.")
col1, col2 = st.columns(2)

# Dropdown menu to choose an option
with col1:
    model_select = st.selectbox(
            "What machine learning model would you like to use?",
            ("Logistic Regression", "Support Vector Machine", "XGBoost", "Naive Bayes", "Random Forest"),
            help="""**Machine Learning Models** is what the program is built around. They are models that are pretrained 
            to find paterns and make decisions from previous data on our binary classification.
            [Learn More](https://www.databricks.com/glossary/machine-learning-models)  \n\n**Binary Classification** 
             is used to categorize the data to either true or fake.
             [Learn More](https://www.learndatasci.com/glossary/binary-classification/)  \n
             \n  **Logistic Regression**, a simple and interpretable binary classification model, assesses the 
             probability of text inputs being fake or real. Valuable for transparently analyzing features. 
             [Learn More](https://towardsdatascience.com/logistic-regression-for-binary-classification-56a2402e62e6)  
             \n*The model accuracy for BoW is* 93% *and for TFIDF the accuracy is* 94%  \n\n  **Support Vector Machine** Support Vector Machines (SVMs) 
             are effective for binary classification, and are good in separating classes in high-dimensional spaces. 
             They handle non-linear text data relationships using kernel tricks and may offer robust generalization 
             in fake news detection. [Learn More](https://towardsdatascience.com/support-vector-machines-for-classification-fc7c1565e3)
             \n*The model accuracy for BoW is* 93% *and for TFIDF the accuracy is* 94%  \n\n  **XGBoost**, short for eXtreme Gradient Boosting, 
            iteratively builds models, correcting errors made by previous ones, to create a strong combined model for 
            accurate predictions through a step-by-step approach. [Learn More](https://www.nvidia.com/en-us/glossary/data-science/xgboost/)
            \n*The model accuracy for BoW is* 93% *and for TFIDF the accuracy is* 93%  \n\n **Naive Bayes**  calculates event probability based 
            on independent conditions. In binary fake news classification, it treats words as independent, ignoring 
            context and interactions among words, hence the term "naive." [Learn More](https://towardsdatascience.com/naive-bayes-explained-9d2b96f4a9c0)
            \n*The model accuracy for BoW is* 86% *and for TFIDF the accuracy is* 85%  \n\n  **Random Forest** , an ensemble learning approach, 
            combines multiple decision trees to improve accuracy and reduce overfitting. In fake news detection, it 
            works well with large text datasets. [Learn More](https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/)
            \n*The model accuracy for BoW is* 79% *and for TFIDF the accuracy is* 86%""",
        )

model_select = model_select.replace(" ","")

with col2:
    # Dropdown menu to choose an option
    vec_select = st.selectbox(
        "What vectorizer would you like to use?",
        ("TFIDF", "Bag Of Words"),
        help="""**Vectorizers** convert text into numerical representations for the models to analyze text data more 
        effectively. For you as a user this the choice of vectorization method has implications for model accuracy, 
        computational efficiency, interpretability.  
             \n  **TF-IDF** (Term Frequency-Inverse Document Frequency) transforms text into numerical data by ranking 
             word importance within a document compared to a collection. It is able to identify unique keywords and 
             anomalies. [Learn More](https://towardsdatascience.com/tf-idf-simplified-aba19d5f5530)
             \n**Bag Of Words** Bag of Words turns text into a numerical representation by comparing the text at hand with 
        a dictionary and counting the occurrence of each word of the dictionary in the given text, without considering 
        the order or structure of the sentences. 
        [Learn More](https://towardsdatascience.com/a-simple-explanation-of-the-bag-of-words-model-b88fc4f4971)""",
    )

vec_select = vec_select.replace(" ","")

SentHelp = "The Sentiment Analyzer (VADER from NLTK) assesses text sentiment with positive, neutral, negative, and " \
           "overall emotionality scores (-1 = most negative and +1 = most positive). While proportional scores offer " \
           "a detailed breakdown, the emotionality score provides nuanced, sometimes differing overall sentiment " \
           "evaluations by considering emotion intensity and interplay, making it valuable for quick assessments, " \
           "especially in mixed-sentiment texts. [Learn More](https://medium.com/@mystery0116/nlp-how-does-nltk-vader-calculate-sentiment-6c32d0f5046b)"

keywordHelp = "YAKE (Yet Another Keyword Extractor) is a text analysis tool that identifies essential keywords by " \
              "considering word significance and context, aiding in content summarization and information retrieval." \
              "[Learn More](https://liaad.github.io/yake/)"

if prompt := st.chat_input("Enter the full text of the article you want to analyze here"):
    if prompt:
        # Creates models
        DATA_PATH = "/mount/src/athena_project/models/"
        mod = open(DATA_PATH + model_select + vec_select + ".pkl", 'rb')
        vec = open(DATA_PATH + vec_select + ".pkl", 'rb')
        model = pickle.load(mod)
        vector = pickle.load(vec)

        tokenizer = nlp.tokenizer
        preprocessed_text = optimized_preprocess([prompt])
        X_input = vector.transform(preprocessed_text)

        if model_select == "SupportVectorMachine":
            prediction = model.predict(X_input)[0]
        else:
            prediction = model.predict(X_input)[0]
            probability = model.predict_proba(X_input)

        kw_extractor = yake.KeywordExtractor()
        language = "en"
        max_ngram_size = 3  # The max_ngram_size is limit the word count of the extracted keyword
        deduplication_threshold = 0.9  # The duplication_threshold variable is limit the duplication of words in different keywords. You can set the deduplication_threshold value to 0.1 to avoid the repetition of words in keywords.
        numOfKeywords = 5  # total keyword extracted will be less than and equal to 10
        custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold,
                                                    top=numOfKeywords, features=None)
        keywords = custom_kw_extractor.extract_keywords(prompt)
        keywordOutput = ""
        for i, kw in enumerate(keywords):
            if i < len(keywords) - 1:
                keywordOutput += f"<strong>{kw[0]   }</strong>, "
            else:
                keywordOutput += f"and <strong>{kw[0]}</strong>. "

        sid = SentimentIntensityAnalyzer()
        ss = sid.polarity_scores(prompt)
        neg = round(ss['neg']*100,3)
        pos = round(ss['pos']*100,3)
        neu = round(ss['neu']*100,3)
        comp = ss['compound']

        st.markdown("""
        <style>
        .big-font {
            font-weight: bold;
            font-size:20px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        if model_select == "SupportVectorMachine":
            if prediction == 'true':
                st.markdown(f'<p class="big-font">True or False?</p> ', help="This displays how sure the model "
                                                                             "is about if the article is true or false",
                            unsafe_allow_html=True)
                st.markdown(f'⚠️ The Support Vector Machine model is not able to give a confidence!')
                st.markdown(
                    f':thumbsup: This model thinks the text is true.')
                st.markdown(f'<p class="big-font">Keywords</p>', help=keywordHelp, unsafe_allow_html=True)
                st.markdown(f'The keywords are: {keywordOutput}', unsafe_allow_html=True)
                st.markdown(f'<p class="big-font">Sentiment Analysis</p>', help=SentHelp, unsafe_allow_html=True)
                st.markdown(f'According to the sentiment analysis <span style="color:red">{neg}%</span> of your '
                            f'text is negative, <span style="color:green">{pos}%</span> positive, <span style="color:grey">{neu}%</span> '
                            f'neutral. In total, the emotionality of the text is '
                            f'rated {comp}.', unsafe_allow_html=True)
                st.markdown(f'<p class="big-font">chatGPT Explanation</p>',
                            help="Here we send the information from our algorithm "
                                 "and the full text article to GPT to try and get "
                                 "a clearer and easier to read explanation of why it's true "
                                 "and other important information", unsafe_allow_html=True)
                with st.spinner("Loading..."):
                    toGPT = f"We have an algorithm that classifies whether texts contain false information or not. " \
                            f"This algorithm is certain that the following text does" \
                            f" not contain false information. Searching for keywords we found these to be the top five:" \
                            f" {keywordOutput}, do you see any conflicting topics in these within the keywords?" \
                            f"Making a sentiment analyzis we found {neg}% of the text is negative, {pos}% positive," \
                            f" {neu}% neutral. In total the emotionality of the text is rated {comp} " \
                            f"(the score is normalized between -1 (negative) and +1 (positive)" \
                            f"Explain what makes the article seem trustworthy and what the intent of the text is? " \
                            f"Does it seem to contain any bias? Highlight a few parts of the text when making examples." \
                            f" Clearly inform the user that this is generated with GPT and that they should make their" \
                            f" own conlcusions.  \n\n{prompt}"
                    GPTExplain(toGPT)
            else:
                st.markdown(f'<p class="big-font">True or False?</p> ', help = "This displays how sure the model "
                                                                               "is about if the article is true or false",unsafe_allow_html=True)
                st.markdown(f'⚠️ The Support Vector Machine model is not able to give a confidence!')
                st.markdown(f':thumbsdown: This model thinks the text is false.')
                st.markdown(f'<p class="big-font">Keywords</p>', help = keywordHelp,unsafe_allow_html=True)
                st.markdown(f'The keywords are: {keywordOutput}', unsafe_allow_html=True)
                st.markdown(f'<p class="big-font">Sentiment Analysis</p>', help = SentHelp, unsafe_allow_html=True)
                st.markdown(f'According to the sentiment analysis <span style="color:red">{neg}%</span> of your '
                            f'text is negative, <span style="color:green">{pos}%</span> positive, <span style="color:grey">{neu}%</span>'
                            f' neutral. In total, the emotionality of the text is '
                            f'rated {comp}.', unsafe_allow_html=True)

                st.markdown(f'<p class="big-font">chatGPT Explanation</p>', help = "Here we send the information from our algorithm "
                                                                                   "and the full text article to GPT to try and get "
                                                                                   "a clearer and easier to read explanation of why it's fake "
                                                                                   "and other important information", unsafe_allow_html=True)
                with st.spinner("Loading..."):
                    toGPT = f"We have an algorithm that classifies whether texts contain false information or not. " \
                            f"This algorithm is certain that the following text" \
                            f" does contain false information. Searching for keywords we found these to be the top five:" \
                            f" {keywordOutput}, do you see any conflicting topics in these within the keywords?" \
                            f"Making a sentiment analyzis we found {neg}% of the text is negative, {pos}% positive," \
                            f" {neu}% neutral. In total the emotionality of the text is rated {comp} " \
                            f"(the score is normalized between -1 (negative) and +1 (positive)" \
                            f"Explain what makes the article does not seem trustworthy and what the intent of the text is? " \
                            f"Does it seem to contain any bias? Highlight a few parts of the text when making examples." \
                            f" Clearly inform the user that this is generated with GPT and that they should make their" \
                            f" own conlcusions.  \n\n{prompt}"
                    GPTExplain(toGPT)

        else:
            if prediction == 'true':
                st.markdown(f'<p class="big-font">True or False?</p> ', help="This displays how sure the model "
                                                                             "is about if the article is true or false",
                            unsafe_allow_html=True)

                st.markdown(
                    f':thumbsup: This model thinks the text is true with a confidence of {int((probability[0][1]) * 100)}%')
                st.markdown(f'<p class="big-font">Keywords</p>', help=keywordHelp, unsafe_allow_html=True)
                st.markdown(f'The keywords are: {keywordOutput}', unsafe_allow_html=True)
                st.markdown(f'<p class="big-font">Sentiment Analysis</p>', help=SentHelp, unsafe_allow_html=True)
                st.markdown(f'According to the sentiment analysis <span style="color:red">{neg}%</span> of your '
                            f'text is negative, <span style="color:green">{pos}%</span> positive, <span style="color:grey">{neu}%</span> '
                            f'neutral. In total, the emotionality of the text is '
                            f'rated {comp}.', unsafe_allow_html=True)
                st.markdown(f'<p class="big-font">chatGPT Explanation</p>',
                            help="Here we send the information from our algorithm "
                                 "and the full text article to GPT to try and get "
                                 "a clearer and easier to read explanation of why it's true "
                                 "and other important information", unsafe_allow_html=True)
                with st.spinner("Loading..."):
                    toGPT = f"We have an algorithm that classifies whether texts contain false information or not. " \
                            f"This algorithm is {int((probability[0][1]) * 100)}% certain that the following text does" \
                            f" not contain false information. Searching for keywords we found these to be the top five:" \
                            f" {keywordOutput}, do you see any conflicting topics in these within the keywords?" \
                            f"Making a sentiment analyzis we found {neg}% of the text is negative, {pos}% positive," \
                            f" {neu}% neutral. In total the emotionality of the text is rated {comp} " \
                            f"(the score is normalized between -1 (negative) and +1 (positive)" \
                            f"Explain what makes the article seem trustworthy and what the intent of the text is? " \
                            f"Does it seem to contain any bias? Highlight a few parts of the text when making examples." \
                            f" Clearly inform the user that this is generated with GPT and that they should make their" \
                            f" own conlcusions.  \n\n{prompt}"
                    GPTExplain(toGPT)
            else:
                st.markdown(f'<p class="big-font">True or False?</p> ', help="This displays how sure the model "
                                                                             "is about if the article is true or false",
                            unsafe_allow_html=True)
                st.markdown(
                    f':thumbsdown: This model thinks the text is false with a confidence of {int((probability[0][0]) * 100)}%')
                st.markdown(f'<p class="big-font">Keywords</p>', help=keywordHelp, unsafe_allow_html=True)
                st.markdown(f'The keywords are: {keywordOutput}', unsafe_allow_html=True)
                st.markdown(f'<p class="big-font">Sentiment Analysis</p>', help=SentHelp, unsafe_allow_html=True)
                st.markdown(f'According to the sentiment analysis <span style="color:red">{neg}%</span> of your '
                            f'text is negative, <span style="color:green">{pos}%</span> positive, <span style="color:grey">{neu}%</span>'
                            f' neutral. In total, the emotionality of the text is '
                            f'rated {comp}.', unsafe_allow_html=True)

                st.markdown(f'<p class="big-font">chatGPT Explanation</p>',
                            help="Here we send the information from our algorithm "
                                 "and the full text article to GPT to try and get "
                                 "a clearer and easier to read explanation of why it's fake "
                                 "and other important information", unsafe_allow_html=True)
                with st.spinner("Loading..."):
                    toGPT = f"We have an algorithm that classifies whether texts contain false information or not. " \
                            f"This algorithm is {int((probability[0][0]) * 100)}% certain that the following text" \
                            f" does contain false information. Searching for keywords we found these to be the top five:" \
                            f" {keywordOutput}, do you see any conflicting topics in these within the keywords?" \
                            f"Making a sentiment analyzis we found {neg}% of the text is negative, {pos}% positive," \
                            f" {neu}% neutral. In total the emotionality of the text is rated {comp} " \
                            f"(the score is normalized between -1 (negative) and +1 (positive)" \
                            f"Explain what makes the article does not seem trustworthy and what the intent of the text is? " \
                            f"Does it seem to contain any bias? Highlight a few parts of the text when making examples." \
                            f" Clearly inform the user that this is generated with GPT and that they should make their" \
                            f" own conlcusions.  \n\n{prompt}"
                    GPTExplain(toGPT)
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
"TF-IDF stands for Term Frequency-Inverse Document Frequency. It's a method to represent the importance of words in a text. Imagine you have a library of books, and you want to determine the significance of a word in a specific book. If a word appears frequently in that book (high Term Frequency), it might be important. However, if the same word appears often in many other books in the library (making it common across documents), its uniqueness or significance diminishes."

"In essence, TF-IDF gives words a weight based on how frequently they appear in one text compared to their presence in a collection of texts. Words that are frequent in a specific text but rare in the entire collection get a higher weight, signaling their significance.")

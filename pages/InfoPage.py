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

st.title("About the project")

st.markdown("This site was made as a student project by four master students during the course TIA400, Introduction to "
            "Human-centered AI. The work is collaboration between Gothenburg University and the Research Institute "
            "of Sweden (RISE) for the ATHENA project. The ATHENA project is a EU project between 14 organizations "
            "and 11 countries.  \n\n'The main aim of ATHENA is to contribute to the protection of democratic processes in "
            "the European Union (EU) against foreign information manipulation and interference (FIMI). "
            "The project will do this by a combination of several things: (i) early detection of FIMI campaigns,"
            "(ii) understanding of the short and long-term behavioural and societal effect of FIMI, (iii)"
            "sharing of FIMI-related knowledge among involved stakeholders and (iv) better understanding"
            "of the efficacy of deployed countermeasures before, during, and after FIMI campaign events."
            "A combination of machine-learning algorithms, field studies, and state-of-the-art detection"
            "tools, will extend the solution-space for policymakers, private stakeholders, and civil society"
            "actors to counter FIMI and strengthen their responses to it'  \n\n"
            "Our student project was to develop a FIMI ML classifier with an interactive interface as well as a prototype"
            " leveraging ChatGPT API that helps to understand the EU’s ethical and legislative frameworks.  \n\n"
            "The work was done over 22 days between the 4th of October and 26th of October 2023.  \n\n")

st.markdown(
            "You can contact us via Email: \n\n Imme Bergman: gusbergim@student.gu.se  \n Lisa Müller: guslismu@student.gu.se  "
            "\n Martin Lindén: gusllimlma@student.gu.se  \nTobias Hermansson: gushertoc@student.gu.se")

st.title("Used Dataset")
st.markdown("The dataset contains a collection of 79,000 articles, which are divided into two categories: 'true' and "
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


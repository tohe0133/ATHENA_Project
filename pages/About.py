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
            "Human-centered AI. The work is a collaboration between Gothenburg University and the Research Institute "
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



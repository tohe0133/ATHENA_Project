# ATHENA_Project

## Introduction

The ATHENA_Project is a comprehensive solution aimed at addressing Fake News and Misinformation in the European Union context. Leveraging the power of machine learning, natural language processing, and an intuitive user interface, this project combines a binary fake news classifier and a Chatbot to assist users in understanding EU texts. The project's application is deployed via Streamlit and can be accessed [here](https://athenaproject-rhjxjf6gvs5r5fhpfr8p3g.streamlit.app/Transparency?fbclid=IwAR3Cd7rqWBQJIRO0gVrkiAQFb-HrxsK6m4SPYGvuGqgnPwCfuVeszvwuz3c).

## Project Structure

- **app**: If you just want to access the app click this link [here](https://athenaproject-rhjxjf6gvs5r5fhpfr8p3g.streamlit.app/InfoPage?fbclid=IwAR1cWa8F7rSsQYSZ4CQl1vWAWa6u1DvTIGEN4qOgdGRozbDgry6AYXE3pcY) .
- **preprocessing_and_model_creation**: If you want to recreate the models and also use the Large Language model BERT for the task of binary fake news classification use the preprocessing_and_model_creation folder. It contains Jupyter Notebooks used for data preprocessing, model building, and training.
  
## Setting Up and Running the Project

### Prerequisites

Before running the project, make sure you have the following installed:
- Python 3.6 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/your-username/ATHENA_Project.git
   cd ATHENA_Project
   ```

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

## Recreating the Models

To recreate the machine learning models used in the ATHENA_Project, follow the instructions below:

1. Download the dataset from [here](https://www.kaggle.com/datasets/stevenpeutz/misinformation-fake-news-text-dataset-79k/).

2. Navigate to the `preprocessing_and_model_creation` folder.

3. Run the `Analyzing_Dividing_Data.ipynb` notebook:
   - This notebook is responsible for preprocessing, cleaning, analyzing, and dividing the data into a train and test dataset.
   - Make sure to update the saving paths to your preferred locations.

4. Run the `fakenews_classifiers.ipynb` notebook:
   - This notebook is used to train and save five different models: Linear Regression, Naive Bayes, SVM, Random Forest, and XGBoost.
   - Make sure to update the data path and the saving path to match your file system.

5. Optionally, run the `Training_BERT.ipynb` notebook:
   - This notebook is for training the LLM BERT for the task of binary fake news detection.
   - Paths need to be updated.

6. Use the `Test_bert.ipynb` file to test the BERT model:
   - Make sure to update paths to match your file system.
     

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries, please open an issue in the GitHub repository or contact the project maintainers directly (Contact info can be found on the infopage of the app).

---

Best Regards,
The ATHENA_Project Team

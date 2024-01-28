# A Web Application to Display Sentiments of Pre-release Tweets on Main Series Pokémon Games

The repository contains the main dataset of the project and the web app. 

To run the web app there are 2 options:

## 1. Run the web app online

Open the link in which the web application is hosted on Streamlit’s cloud’s service:
https://pokemon-sentiment.streamlit.app/

## 2. Run the web app locally

Install Streamlit from this reference:
https://docs.streamlit.io/library/get-started/installation

On the terminal, run this code:
>`pip install streamlit`

Once installed follow this reference: https://docs.streamlit.io/knowledge-base/using-streamlit/how-do-i-run-my-streamlit-script
	
In the folder where ”streamlit_app.py” is located, open the terminal and run this code:

>`streamlit run streamlit_app.py`

---
The following are the contents of the repository:
- In the main folder, there are ”fig_all.pkl” which is a pickled file of a Plotly object which shows the number of tweets per day for all games, ”requirements.txt” which is the requirements for the app to run, and ”streamlit_app.py” which is the main web app python script.
- The data of all tweets and sentiment is inside the ”datasets” folder.
- The metadata for the web app is contained in the ”metadata” folder which includes the user guide that's built in to the web app as well as the descriptions of the games.
---

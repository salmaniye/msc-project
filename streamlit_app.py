import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import io
import altair as alt
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

st.set_page_config(layout="wide")

# initializing containers
dataset, header = st.tabs(["Data Plots", "About"])
inputs = st.form(key='form',clear_on_submit=True)
col1, col2 = st.columns(2)
common = st.container()

# functions
@st.experimental_memo
def call_dataset(game_name):
	# load data
	game_csv = game_name
	data = pd.read_csv(f'datasets/{game_csv}_tweets_datatset.csv',lineterminator='\n')

	data = data[data['text'].str.contains('I liked a YouTube')== False]
	data = data[data['text'].str.contains('I liked a @YouTube video')== False]
	data = data[data['text'].str.contains('I added a video to a @YouTube')== False]
	data = data[data['text'].str.contains('I added a video to a YouTube')== False]
	data = data[data['text'].str.contains('Giveaway')== False]

	# removing \r in 'sentiment'
	if 'sentiment\r' in data.columns:
		data['sentiment'] = data['sentiment\r'].apply(lambda x: x.replace('\r',''))
		data.drop(columns=['sentiment\r'],inplace=True)
	
	# removing first columns
	data.drop(data.columns[0], axis=1, inplace=True)

	# changing 'sentiment scores' from str to ndarray
	data['sentiment scores'] = data['sentiment scores'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

	# changing 'tweet id' from float64 to str
	data['tweet id'] = data['tweet id'].apply(lambda x: str(int(x)))

	# adding sentiment scores
	sentiment_score = []
	sentiment_confidence = []
	for score in data['sentiment scores']:
		sentiment_score.append(score.argmax())
		sentiment_confidence.append(score[score.argmax()])
	
	data['sentiment score'] = sentiment_score
	data['sentiment confidence'] = sentiment_confidence

	# changing 'created at' date from str to datetime	
	data['created at'] = data['created at'].apply(lambda x: x.removesuffix('+00:00'))
	data['created at'] = data['created at'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
	data['date'] = data['created at'].apply(lambda x: datetime.date(x))
	
	# sorting from earliest to latest
	data.sort_values(by='created at', inplace=True)
	data.reset_index(drop=True, inplace=True)
	return data

@st.experimental_memo
def func_sentiment_per_day(df):
	df = df.groupby(['sentiment','date'], as_index=False).size()
	return df

@st.experimental_memo
def func_slider_df_size(df,date_range):
	df = df[df['date'].between(date_range[0],date_range[1],inclusive=True)]
	return df

@st.experimental_memo
def func_slider_df_all(df,date_range):
	df = df[df['date'].between(date_range[0],date_range[1],inclusive=True)]
	return df

@st.experimental_memo
def func_filtered_df(df,options_sentiment):
	df = df[df["sentiment"].isin(options_sentiment)]
	return df

@st.experimental_memo
def func_keyword(df,key):
	df =  df[df['text'].str.contains(pat=key, case=False)==True]
	return df

with header:
	st.title("Sentiment of gamers' pre-release tweets on main series Pokémon games")
	st.markdown("""This is a web app that displays tweets and their sentiment on the selected Pokémon game.

place holder""")


with dataset:
	st.caption('**Please note the control panel is on the sidebar**')
	games_list = ['Pokémon X&Y', 'Pokémon Omega Ruby & Alpha Sapphire',
				  'Pokémon Sun & Moon', 'Pokémon Ultra Sun & Ultra Moon',
				  "Pokémon Let's Go, Pikachu! and Let's Go, Eevee!",
				  'Pokémon Sword & Shield', 'Pokémon Sword and Shield: The Isle of Armor and The Crown Tundra',
				  'Pokémon Brilliant Diamond & Shining Pearl',
				  'Pokémon Legends: Arceus', 'Pokémon Scarlet & Violet']

	games_csv = ['xy','oras','sunmoon','ultrasm','letsgo','swsh','swshdlc','bdsp','arceus','sv']
	games_zip = zip(games_list,games_csv)
	games_dict = dict(games_zip)

	# a dropdown for the user to choose the game to be displayed
	### SIDEBAR
	st.sidebar.markdown("# Control Panel")
	game_name = st.sidebar.selectbox('Select a game to display:', games_list)
	st.sidebar.caption(f'You have selected: {game_name}')

	game_dataset = call_dataset(games_dict[game_name])

	# grouping sentiment per date
	sentiment_per_day = func_sentiment_per_day(game_dataset)
	min_date = sentiment_per_day['date'].min()
	max_date = sentiment_per_day['date'].max()

	# initializing session state
	if 'dateinput1' not in st.session_state:
		st.session_state['dateinput1'] = min_date
		st.session_state['dateinput2'] = max_date
		st.session_state['opsentiment'] = ['Positive', 'Neutral', 'Negative']
		st.session_state['kw_s'] = ""

	# date input
	### SIDEBAR
	# date_range = st.sidebar.slider('Please select the range of dates:', min_date, max_date, (min_date, max_date))
	date_range = list([0,0])

with inputs:
	with st.sidebar:
		date_range[0] = st.date_input('Select starting date:', min_date, min_date, max_date,key='dateinput1')
		date_range[1] = st.date_input('Select end date:', max_date,date_range[0],max_date,key='dateinput2')

		options_sentiment = st.multiselect(label='Filter by sentiment (dropdown):',
			options=['Positive', 'Neutral', 'Negative'],
			default=['Positive', 'Neutral', 'Negative'],
			key='opsentiment')

		keyword_text = st.text_input('Search text within the date range (case insensitive):', key='kw_s')
		if keyword_text:
			st.caption(f'The current text search is: {keyword_text}')
		else:
			st.caption(f'No text search input')

		# submit button
		submitted = st.form_submit_button("Submit")
		if submitted:
			st.write("Submitted")

def clear_inputs():
	st.session_state['dateinput1'] = min_date
	st.session_state['dateinput2'] = max_date
	st.session_state['opsentiment'] = ['Positive', 'Neutral', 'Negative']
	st.session_state['kw_s'] = ""
	return

with dataset:
	# # check state
	# st.session_state

	#create your button to clear the state of the multiselect
	st.sidebar.button("Reset Values", on_click=clear_inputs)

	if keyword_text:
		game_dataset = func_keyword(game_dataset,keyword_text)

	sentiment_per_day = func_sentiment_per_day(game_dataset )
	slider_df = func_slider_df_size(sentiment_per_day,date_range)

	
	# creates a dataframe of tweets created between dates chosen
	date_range_df = func_slider_df_all(game_dataset,date_range)
	st.text(f'Tweets from {date_range[0]} to {date_range[1]}')

	game_dataset_clean = date_range_df[['text','date','sentiment scores','sentiment']]

	filtered_df = func_filtered_df(game_dataset_clean,options_sentiment)
	# search text in dataframe
	if keyword_text:
		filtered_df = func_keyword(filtered_df,keyword_text)

	# fig1. sentiment over time
	st.markdown(f"<h5 style='text-align: center;'>Sentiment on {game_name} over time</h5>", unsafe_allow_html=True)
	# st.subheader(f"Sentiment on {game_name} over time")
	slider_df = slider_df[slider_df["sentiment"].isin(options_sentiment)]

	fig = px.line(slider_df, x='date', y='size', labels={
		'date':'Date',
		'size':'Number of tweets',
		'sentiment':'Sentiment'},
		color='sentiment',
		color_discrete_map={'Negative':'#DC3912','Neutral':'#3366CC','Positive':'#109618'}) 
		#['red', 'blue', 'green']
	fig.update_layout(title_text="Number of tweets and their sentiment over time", title_x=0.5)
	with col1:
		st.write(fig)

	# fig2. normalized sentiment area over time
	sentiment_total_pd = slider_df.groupby(['date'], as_index=False).sum()
	spd = slider_df.merge(sentiment_total_pd, left_on = 'date', right_on='date')
	spd['sentiment percentage'] = spd['size_x']/spd['size_y']
	fig2 = px.area(spd, x='date', y='sentiment percentage',labels={
		'date':'Date',
		'sentiment percentage':'Sentiment (%)',
		'sentiment':'Sentiment'},
		color='sentiment',
		color_discrete_map={'Negative':'#DC3912','Neutral':'#3366CC','Positive':'#109618'},
		category_orders={"sentiment": ["Negative", "Neutral", "Positive"]})
	fig2.update_layout(title_text="Normalized sentiment over time", title_x=0.5)
	with col2:
		st.write(fig2)

with common:
	# display tweets
	st.markdown(f"<h5 style='text-align: center;'>Tweets on {game_name}</h5>", unsafe_allow_html=True)
	st.dataframe(filtered_df)

	dataset_text = ' '.join(game_dataset['preprocessed tweets'])

	# remove_words = ['https', 'Pokémon', 'pokemon','Pokemon', 'POKEMON','amp','t','co','RT',
	# 				'X','Y','x','y','Sun','Moon','SunMoon','PokemonSunMoon',
	# 				'Alpha','Sapphire','Omega', 'Ruby','ORAS',
	# 				'user']

	# for word in remove_words:
	# 	dataset_text = dataset_text.replace(word,'')

	fig_word, ax = plt.subplots()

	wordcloud = WordCloud(background_color='white', colormap='Set2',
				collocations=False, stopwords = STOPWORDS).generate(dataset_text)

	st.sidebar.header('Word cloud of most common words')
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.show()
	st.sidebar.pyplot(fig_word)

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
with st.sidebar:
	input_container, word_cloud,  header = st.tabs(["Control Panel", "WordCloud", "About"])

dataset = st.container()
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
	st.title("Sentiment of pre-release tweets on main series Pokémon games")
	st.markdown("""This is a web app that displays tweets and their sentiment on the selected Pokémon game.

place holder""")

games_list = ['Pokémon X&Y', 'Pokémon Omega Ruby & Alpha Sapphire',
			  'Pokémon Sun & Moon', 'Pokémon Ultra Sun & Ultra Moon',
			  "Pokémon: Let's Go, Pikachu! and Let's Go, Eevee!",
			  'Pokémon Sword & Shield', 'Pokémon Sword and Shield: The Isle of Armor and The Crown Tundra',
			  'Pokémon Brilliant Diamond & Shining Pearl',
			  'Pokémon Legends: Arceus', 'Pokémon Scarlet & Violet']

games_csv = ['xy','oras','sunmoon','ultrasm','letsgo','swsh','swshdlc','bdsp','arceus','sv']
games_zip = zip(games_list,games_csv)
games_dict = dict(games_zip)

with input_container:
	# a dropdown for the user to choose the game to be displayed
	# st.markdown("# Control Panel")
	game_name = st.selectbox('Select a game to display:', games_list)
	st.caption(f'You have selected: {game_name}')

	with st.expander(f"About {game_name}"):
		if game_name == 'Pokémon X&Y':
			st.image("https://oyster.ignimgs.com/mediawiki/apis.ign.com/pokemon-x-y-version/2/2c/Pokemon_xy_box_art.png")
		if game_name == 'Pokémon Omega Ruby & Alpha Sapphire':
			st.image("https://oyster.ignimgs.com/mediawiki/apis.ign.com/pokemon-omega-ruby-and-alpha-sapphire/2/20/Pokemon-oras-box-art.png?width=1280")
		if game_name == 'Pokémon Sun & Moon':
			st.image("https://nintendoeverything.com/wp-content/uploads/pokemon-ultra-sun-ultra-moon-boxart.jpg")
		if game_name == 'Pokémon Ultra Sun & Ultra Moon':
			st.image("https://nintendoeverything.com/wp-content/uploads/pokemon-ultra-sun-ultra-moon-boxart.jpg")
		if game_name == "Pokémon: Let's Go, Pikachu! and Let's Go, Eevee!":
			st.image("https://projectpokemon.org/home/uploads/monthly_2019_06/large.pikavee.png.d7bf0a83bc6ff9577002fdd2d8d5d68c.png")
		if game_name == 'Pokémon Sword & Shield':
			st.image("https://projectpokemon.org/home/uploads/monthly_2019_06/large.swordnshield.png.051ae8b21788af441c7493b55ff8ad85.png")
		if game_name == "Pokémon Sword and Shield: The Isle of Armor and The Crown Tundra":
			st.image("https://i0.wp.com/www.rapidreviewsuk.com/wp-content/uploads/2020/06/Pokemon-DLC-Header-e1593108081197.png?fit=1301%2C533&ssl=1")
		if game_name == 'Pokémon Brilliant Diamond & Shining Pearl':
			st.image("https://d28ipuewd7cdcq.cloudfront.net/assets/editorial/2021/05/pokemon-brilliant-diamond-pokemon-shining-pearl-box-art.png")
		if game_name == 'Pokémon Legends: Arceus':
			st.image("https://m.media-amazon.com/images/I/71HYKF4rO9L._AC_SY445_.jpg")
		if game_name == 'Pokémon Scarlet & Violet':
			st.image("https://img.buzzfeed.com/buzzfeed-static/static/2022-06/1/14/asset/67720d6b2786/sub-buzz-3952-1654093639-8.png?downsize=700%3A%2A&output-quality=auto&output-format=auto")


game_dataset = call_dataset(games_dict[game_name])

with input_container:
	inputs = st.form(key='form',clear_on_submit=False)

# grouping sentiment per date
sentiment_per_day = func_sentiment_per_day(game_dataset)
min_date = sentiment_per_day['date'].min()
max_date = sentiment_per_day['date'].max()

date_range = list([0,0])

# function for clearing inputs by restting session states
def clear_inputs():
	st.session_state['dateinput1'] = min_date
	st.session_state['dateinput2'] = max_date
	st.session_state['opsentiment'] = ['Positive', 'Neutral', 'Negative']
	st.session_state['kw_s'] = ""
	return

with inputs:
	# start and end dates
	date_range[0] = st.date_input('Select starting date:', min_date, min_date, max_date,key='dateinput1')
	date_range[1] = st.date_input('Select end date:', max_date,date_range[0],max_date,key='dateinput2')

	# options for filtering sentiment
	options_sentiment = st.multiselect(label='Filter by sentiment (dropdown):',
		options=['Positive', 'Neutral', 'Negative'],
		default=['Positive', 'Neutral', 'Negative'],
		key='opsentiment')

	# search text in dataframe
	keyword_text = st.text_input('Search text within the date range (case insensitive):', key='kw_s')
	if keyword_text:
		st.caption(f'The current text search is: {keyword_text}')
	else:
		st.caption(f'No text search input')

	# submit button
	submitted = st.form_submit_button("Submit")

	if submitted:
		st.write("Submitted")
	#create your button to clear the state of the multiselect


with input_container:
	st.button("Reset options to default values", on_click=clear_inputs)


if keyword_text:
	game_dataset = func_keyword(game_dataset,keyword_text)

# dataframe for number of tweets
sentiment_per_day = func_sentiment_per_day(game_dataset)
slider_df = func_slider_df_size(sentiment_per_day,date_range)
slider_df = slider_df[slider_df["sentiment"].isin(options_sentiment)]

# creates a dataframe of tweets created between dates chosen
date_range_df = func_slider_df_all(game_dataset,date_range)
game_dataset_clean = date_range_df[['text','date','sentiment scores','sentiment']]
filtered_df = func_filtered_df(game_dataset_clean,options_sentiment)
if keyword_text:
	filtered_df = func_keyword(filtered_df,keyword_text)

# fig1. sentiment over time
fig = px.line(slider_df, x='date', y='size', labels={
	'date':'Date',
	'size':'Number of tweets',
	'sentiment':'Sentiment'},
	color='sentiment',
	color_discrete_map={'Positive':'#109618','Neutral':'#3366CC','Negative':'#DC3912'}) 
	#['red', 'blue', 'green']
fig.update_layout(title_text="Number of tweets and their sentiment over time", title_x=0.5)

# fig2. normalized sentiment area over time
sentiment_total_pd = slider_df.groupby(['date'], as_index=False).sum()
spd = slider_df.merge(sentiment_total_pd, left_on = 'date', right_on='date')
spd['sentiment percentage'] = spd['size_x']/spd['size_y']
fig2 = px.area(spd, x='date', y='sentiment percentage',labels={
	'date':'Date',
	'sentiment percentage':'Sentiment (%)',
	'sentiment':'Sentiment'},
	color='sentiment',
	color_discrete_map={'Positive':'#109618','Neutral':'#3366CC','Negative':'#DC3912'},
	category_orders={"sentiment": ["Negative", "Neutral", "Positive"]})
fig2.update_layout(title_text="Normalized sentiment over time", title_x=0.5)

with dataset:
	st.markdown(f"<h3 style='text-align: center;'>Sentiment of pre-release tweets on main series Pokémon games</h3>", unsafe_allow_html=True)
	st.caption('**Please note the control panel is on the sidebar**')
	st.text(f'Tweets from {date_range[0]} to {date_range[1]}')
	with col1:
		st.write(fig)
	with col2:
		st.write(fig2)

with common:
	# display tweets
	st.markdown(f"<h5 style='text-align: left;'>Tweets on {game_name}</h5>", unsafe_allow_html=True)
	st.dataframe(filtered_df)

# with word_cloud:
# 	dataset_text = ' '.join(game_dataset['preprocessed tweets'])
# 	# dataset_text = ''.join(ch for ch in string_value if ch.isalnum())
# 	# remove_words = ['https', 'Pokémon', 'pokemon','Pokemon', 'POKEMON','amp','t','co','RT',
# 	# 				'X','Y','x','y','Sun','Moon','SunMoon','PokemonSunMoon',
# 	# 				'Alpha','Sapphire','Omega', 'Ruby','ORAS',
# 	# 				'user']

# 	# for word in remove_words:
# 	# 	dataset_text = dataset_text.replace(word,'')

# 	fig_word, ax = plt.subplots()

# 	wordcloud = WordCloud(background_color='white', colormap='Set2',
# 				collocations=False, stopwords = STOPWORDS).generate(dataset_text)

# 	st.caption('Word cloud of most common words between the date range and text search')
# 	plt.imshow(wordcloud, interpolation='bilinear')
# 	plt.axis("off")
# 	plt.show()
# 	st.pyplot(fig_word)

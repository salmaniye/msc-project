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

header = st.container()
dataset = st.container()
plots = st.container()
kw_search = st.container()
common = st.container()

#@st.cache
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

with header:
	st.title("Sentiment of gamers' pre-release tweets on main series Pokémon games")
	st.markdown('The main series of Pokémon games are one of the highest selling video games of all time')

with dataset:
	st.header('Tweets on Pokémon games')

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
	game_name = st.selectbox('Select a game to display:', games_list)

	st.write('You have selected:', game_name)

	game_dataset = call_dataset(games_dict[game_name])

	# grouping sentiment per date
	sentiment_per_day = game_dataset.groupby(['sentiment','date'], as_index=False).size()
	min_date = sentiment_per_day['date'].min()
	max_date = sentiment_per_day['date'].max()
	date_range = st.slider('date range', min_date, max_date, (min_date, max_date))

	# add a slider for user to input
	slider_df = sentiment_per_day[sentiment_per_day['date'].between(date_range[0],date_range[1],inclusive=True)]

with plots:
	# creates a dataframe of tweets created between dates chosen
	date_range_df = game_dataset[game_dataset['date'].between(date_range[0],date_range[1],inclusive=True)]
	st.text(f'Tweets from {date_range[0]} to {date_range[1]}')
	game_dataset_clean = date_range_df[['text','date','sentiment scores','sentiment']]
	st.dataframe(game_dataset_clean)

	# fig1. sentiment per day
	st.header(f"Sentiment on {game_name} per day")
	fig = px.line(slider_df, x='date', y='size', title='Number of tweets their sentiment over time', 
		color='sentiment', color_discrete_sequence=['#DC3912', '#3366CC', '#109618']) #['red', 'blue', 'green']
	st.write(fig)

	# fig2. normalized sentiment area per day
	sentiment_total_pd = sentiment_per_day.groupby(['date'], as_index=False).sum()
	spd = sentiment_per_day.merge(sentiment_total_pd, left_on = 'date', right_on='date')
	spd['sentiment percentage'] = spd['size_x']/spd['size_y']
	fig2 = px.area(spd, x='date', y='sentiment percentage', title='Normalized sentiment per day',
				 color='sentiment', color_discrete_sequence=['#DC3912', '#3366CC', '#109618'])
	st.write(fig2)

	# area plot

	# fig_area = px.area(slider_df, x='date', y='size', title='sentiment area per day',
	# 	color='sentiment', color_discrete_sequence=['#DC3912', '#3366CC', '#109618']) #['red', 'blue', 'green']
	
	# st.write(fig_area)

with kw_search:

	# search keyword in dataframe
	keyword = st.text_input('Search keyword within the date range (case insensitive):')
	st.write('The current keyword is:', keyword)

	keyword_df = date_range_df[date_range_df['text'].str.contains(pat=keyword, case=False)==True]
	st.dataframe(keyword_df)

	keyword_per_day = keyword_df.groupby(['sentiment','date'], as_index=False).size()
	fig_kw = px.line(slider_df, x='date', y='size', title=f'Number of tweets with "{keyword}" and their sentiment over time', 
		color='sentiment', color_discrete_sequence=['#DC3912', '#3366CC', '#109618']) #['red', 'blue', 'green']
	st.write(fig_kw)

with common:
	# most common words

	most_common_words = Counter(" ".join(game_dataset["text"]).split()).most_common(100)

	dataset_text = ' '.join(game_dataset['preprocessed tweets'])
	# remove_words = ['https', 'Pokémon', 'pokemon','Pokemon', 'POKEMON','amp','t','co','RT',
	# 				'X','Y','x','y','Sun','Moon','SunMoon','PokemonSunMoon',
	# 				'Alpha','Sapphire','Omega', 'Ruby','ORAS',
	# 				'user']

	# for word in remove_words:
	# 	dataset_text = dataset_text.replace(word,'')

	fig_word, ax = plt.subplots()
	wordcloud = WordCloud(background_color='white', colormap='Set2', collocations=False, stopwords = STOPWORDS).generate(dataset_text)

	st.header('Word cloud of most common words')
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.show()
	st.pyplot(fig_word)
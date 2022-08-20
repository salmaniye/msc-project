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

st.caption('**Please note the control panel is on the sidebar**')

# initializing containers
dataset, kw_search, header = st.tabs(["Data Plots", "Text Search", "About"])
common = st.container()
inputs = st.form(key='form',clear_on_submit=True)

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
					
With over 60 million copies sold in the past fiscal year (FY2022) alone, Pokémon games are amongst the most well-known games in the gaming industry. Sentiment analysis of the hype surrounding these games offers a wealth of information that may be helpful towards advertisers and retailers. 

For advertisers, sentiment analysis is a great tool for deciding which games to market and promote, it can assist with brand monitoring by examining both the quantity and quality of brand mentions. Because the intended game to be advertised is Pokémon, a multibillion-dollar franchise, the popularity of the games alone guarantees millions of sales, however if the general sentiment for the new games is negative or there is not enough positive hype towards the games then sales numbers can be significantly impacted. Using the data, advertisers can determine when they should advertise to regenerate hype.

Before a game is released, retailers can learn how the game is being accepted through sentiment analysis. When pre-orders are open, if the anticipation and sentiment surrounding the game is good, retailers and e-commerce websites can highlight or promote it to secure more sales. If a game receives negative reactions, they can remove it from their list of suggestions or promote a better anticipated game to maximize those sales.

Additionally, game developers can also use this data to determine the popularity and sentiment of a feature or game mechanic they have revealed by searching for keywords in the tweets. This lets them to choose which mechanic to focus on, where to improve, and even what to scrap. 

Using a pre-trained sentiment analysis model trained on tweets, sentiment analysis can help the professionals involved in making and selling video games make better products, satisfying both customers and shareholders.
""")


with dataset:
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
	
	st.header(f'Tweets on {game_name}')
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

	# add a slider for user to input
	### SIDEBAR
	# date_range = st.sidebar.slider('Please select the range of dates:', min_date, max_date, (min_date, max_date))
	date_range = list([0,0])
	# date_range[0] = st.sidebar.date_input('Select starting date:', min_date, min_date, max_date)
	# date_range[1] = st.sidebar.date_input('Select end date:', max_date,date_range[0],max_date)
with inputs:
	with st.sidebar:
		date_range[0] = st.date_input('Select starting date:', min_date, min_date, max_date,key='dateinput1')
		date_range[1] = st.date_input('Select end date:', max_date,date_range[0],max_date,key='dateinput2')

		options_sentiment = st.multiselect(label='Filter by sentiment (dropdown):',
			options=['Positive', 'Neutral', 'Negative'],
			default=['Positive', 'Neutral', 'Negative'],
			key='opsentiment')

		submitted = st.form_submit_button("Submit")
		if submitted:
			st.write("Submitted")

def clear_inputs():
	st.session_state['dateinput1'] = min_date
	st.session_state['dateinput2'] = max_date
	st.session_state['opsentiment'] = ['Positive', 'Neutral', 'Negative']
	return

with dataset:
	# # check state
	# st.session_state

	#create your button to clear the state of the multiselect
	st.sidebar.button("Reset Values", on_click=clear_inputs)

	slider_df = func_slider_df_size(sentiment_per_day,date_range)

	# creates a dataframe of tweets created between dates chosen
	date_range_df = func_slider_df_all(game_dataset,date_range)
	st.text(f'Tweets from {date_range[0]} to {date_range[1]}')
	game_dataset_clean = date_range_df[['text','date','sentiment scores','sentiment']]

	filtered_df = func_filtered_df(game_dataset_clean,options_sentiment)
	st.dataframe(filtered_df)

	# fig1. sentiment over time
	st.header(f"Sentiment on {game_name} over time")
	slider_df = slider_df[slider_df["sentiment"].isin(options_sentiment)]
	fig = px.line(slider_df, x='date', y='size', labels={
		'date':'Date',
		'size':'Number of tweets',
		'sentiment':'Sentiment'},
		title='Number of tweets and their sentiment over time', color='sentiment',
		color_discrete_map={'Negative':'#DC3912','Neutral':'#3366CC','Positive':'#109618'}) 
		#['red', 'blue', 'green']
	st.write(fig)

	# fig2. normalized sentiment area over time
	sentiment_total_pd = slider_df.groupby(['date'], as_index=False).sum()
	spd = slider_df.merge(sentiment_total_pd, left_on = 'date', right_on='date')
	spd['sentiment percentage'] = spd['size_x']/spd['size_y']
	fig2 = px.area(spd, x='date', y='sentiment percentage',labels={
		'date':'Date',
		'sentiment percentage':'Sentiment (%)',
		'sentiment':'Sentiment'},
		title='Normalized sentiment over time', color='sentiment',
		color_discrete_map={'Negative':'#DC3912','Neutral':'#3366CC','Positive':'#109618'})
	st.write(fig2)

# with kw_search:

# 	# search text in dataframe
# 	### SIDEBAR
# 	keyword_text = st.text_input('Search text within the date range (case insensitive):')
# 	if keyword_text:
# 		st.caption(f'The current text search is: {keyword_text}')
# 	else:
# 		st.caption(f'No text search input')

# 	if keyword_text:
# 		keyword_df = func_keyword(date_range_df,keyword_text)
# 		st.header('Text Search')
# 		st.write(f'Tweets with "{keyword_text}"')
# 		st.dataframe(keyword_df[['text','date','sentiment scores','sentiment']])

# 		keyword_per_day = keyword_df.groupby(['sentiment','date'], as_index=False).size()
# 		filtered_kpd = func_filtered_df(keyword_per_day,options_sentiment)
# 		fig_kw = px.line(filtered_kpd, x='date', y='size',
# 			title=f'Number of tweets with "{keyword_text}" and their sentiment over time', color='sentiment',
# 			color_discrete_map={'Negative':'#DC3912','Neutral':'#3366CC','Positive':'#109618'})
# 			#['red', 'blue', 'green']
# 		st.write(fig_kw)

# 		k_total_pd = filtered_kpd.groupby(['date'], as_index=False).sum()
# 		kpd = filtered_kpd.merge(k_total_pd, left_on = 'date', right_on='date')
# 		kpd['sentiment percentage'] = kpd['size_x']/kpd['size_y']
# 		fig_kw2 = px.area(kpd, x='date', y='sentiment percentage',labels={
# 			'date':'Date',
# 			'sentiment percentage':'Sentiment (%)',
# 			'sentiment':'Sentiment'},
# 			title='Normalized sentiment over time', color='sentiment',
# 			color_discrete_map={'Negative':'#DC3912','Neutral':'#3366CC','Positive':'#109618'})
# 		st.write(fig_kw2)

with common:
	dataset_text = ' '.join(date_range_df['preprocessed tweets'])

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

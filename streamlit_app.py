import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import altair as alt
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import json

st.set_page_config(layout="wide")
about = st.container()
# initializing containers
with st.sidebar:
	input_container, word_cloud, help_tab = st.tabs(["Control Panel", "WordCloud", "Help"])

metrics = st.container()
with metrics:
	st.caption('**Please refer to "Help" on the sidebar on how to use the web app**')
	st.markdown(f'Overall metrics difference from previous game (this is unaffected by options):')
	subcol1, subcol2, subcol3 = st.columns(3)
	st.markdown(f'***')

dataset = st.container()
col1, col2 = st.columns(2)
common = st.container()

# loading metadata
with open(f"datasets/games_metadata.json") as file:
    game_metadata = json.load(file)

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
	# for creating df with number of tweets
	df = df.groupby(['sentiment','date'], as_index=False).size()
	return df

@st.experimental_memo
def func_slider_df_size(df,date_range):
	# for filtering dates of df with number of tweets
	df = df[df['date'].between(date_range[0],date_range[1],inclusive='both')]
	return df

@st.experimental_memo
def func_slider_df_all(df,date_range):
	# for filtering dates of df with original columns
	df = df[df['date'].between(date_range[0],date_range[1],inclusive='both')]
	return df

@st.experimental_memo
def func_filtered_df(df,options_sentiment):
	# for filtering sentiment
	df = df[df["sentiment"].isin(options_sentiment)]
	return df

@st.experimental_memo
def func_keyword(df,key):
	# for filtering with text search
	df =  df[df['text'].str.contains(pat=key, case=False)==True]
	return df

with about:
	st.markdown(f"## Sentiment of pre-release tweets on main series Pokémon games")
	st.markdown("""This is a web app that displays tweets and their sentiment on the selected Pokémon game.""")

with help_tab:
	st.markdown("""# User Guide:

The control panel provides options to choose which game to display, starting and end dates, filtering by sentiment, and text search.

***

## Control Panel
Game selection:
- Press anywhere within the select widget to view list of games.

Starting and end dates:
- Choose from the calendar the starting and end dates
- String search is also possible, just type in the box

Filtering by sentiment:
- Press the (x) to remove sentiment to filter

Text search:
- Type anything into the search box

To submit all of the above selections, press the **Submit** button.

***

## Figures
Table Interactivity:
- Column sorting: sort columns by clicking on their headers.
- Column resizing: resize columns by dragging and dropping column header borders.
- Table (height, width) resizing: resize tables by dragging and dropping the bottom right corner of tables.
- Search: search through data by clicking a table, using hotkeys (⌘ Cmd + F or Ctrl + F) to bring up the search bar, and using the search bar to filter data.
- Copy to clipboard: select one or multiple cells, copy them to clipboard, and paste them into your favorite spreadsheet software.""")
	
	st.caption("[*Source*](https://docs.streamlit.io/library/api-reference/data/st.dataframe)")
	st.markdown("""***

All data, including the app, is stored on [this GitHub repository]""") #(https://github.com/salmaniye/msc-project)

	st.caption('by Salman Fatahillah/sxf181')

games_list = ['Pokémon X&Y', 'Pokémon Omega Ruby & Alpha Sapphire',
			  'Pokémon Sun & Moon', 'Pokémon Ultra Sun & Ultra Moon',
			  "Pokémon: Let's Go, Pikachu! and Let's Go, Eevee!",
			  'Pokémon Sword & Shield', 'Pokémon Sword and Shield: The Isle of Armor and The Crown Tundra',
			  'Pokémon Brilliant Diamond & Shining Pearl',
			  'Pokémon Legends: Arceus', 'Pokémon Scarlet & Violet']

games_csv = ['xy','oras','sunmoon','ultrasm','letsgo','swsh','swshdlc','bdsp','arceus','sv']
games_zip = zip(games_list,games_csv)
games_dict = dict(games_zip)
games_dict_index = dict(zip(games_list,range(10)))

metrics_data = pd.read_csv(f'datasets/games_metrics.csv')

########################################################################################################
def call_metric_data(index_no):
	if index_no == 0:
		with subcol1:
			st.metric(label="Positive", value=f'{metrics_data.loc[0][1]:.2f}%', delta=None)
		with subcol2:
			st.metric(label="Neutral", value=f'{metrics_data.loc[0][2]:.2f}%', delta=None)
		with subcol3:
			st.metric(label="Negative", value=f'{metrics_data.loc[0][3]:.2f}%', delta=None)
	else:
		with subcol1:
			delta1 = metrics_data.loc[index_no][1] - metrics_data.loc[index_no-1][1]
			st.metric(label="Positive", value=f'{metrics_data.loc[index_no][1]:.2f}%', delta=f'{delta1:.2f}%')
		with subcol2:
			delta2 = metrics_data.loc[index_no][2] - metrics_data.loc[index_no-1][2]
			st.metric(label="Neutral", value=f'{metrics_data.loc[index_no][2]:.2f}%', delta=f'{delta2:.2f}%')
		with subcol3:
			delta3 = metrics_data.loc[index_no][3] - metrics_data.loc[index_no-1][3]
			st.metric(label="Negative", value=f'{metrics_data.loc[index_no][3]:.2f}%', delta=f'{delta3:.2f}')


with input_container:
	# a dropdown for the user to choose the game to be displayed
	# st.markdown("# Control Panel")
	game_name = st.selectbox('Select a game to display:', games_list)
	st.caption(f'You have selected: {game_name}')

	with st.expander(f"About {game_name}"):
		for entry in game_metadata:
			if game_name == entry['name']:
				st.markdown(f"""{entry['announced']}  
					{entry['released']}""")
				st.image(entry['image'])
				st.caption(entry['source'])
				st.markdown(entry['paragraph1'])
				st.markdown(entry['paragraph2'])
				st.markdown(entry['suggested_searches'])
				call_metric_data(games_dict_index[game_name])
########################################################################################################

@st.experimental_memo
def func_creating_fig1(df):
	fig = px.line(df, x='date', y='size', labels={
		'date':'Date',
		'size':'Number of tweets',
		'sentiment':'Sentiment'},
		color='sentiment',
		color_discrete_map={'Positive':'#109618','Neutral':'#3366CC','Negative':'#DC3912'}) 
		#['red', 'blue', 'green']
	fig.update_layout(title_text=f"Number of tweets on {game_name} and their sentiment over time", title_x=0.5)
	return fig

@st.experimental_memo
def func_creating_fig2(df):
	fig2 = px.area(df, x='date', y='sentiment percentage',labels={
		'date':'Date',
		'sentiment percentage':'Sentiment (%)',
		'sentiment':'Sentiment'},
		color='sentiment',
		color_discrete_map={'Positive':'#109618','Neutral':'#3366CC','Negative':'#DC3912'},
		category_orders={"sentiment": ["Negative", "Neutral", "Positive"]})
	fig2.update_layout(title_text=f"Normalized sentiment of tweets on {game_name} over time", title_x=0.5)
	return fig2

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
fig = func_creating_fig1(slider_df)


# fig2. normalized sentiment area over time
sentiment_total_pd = slider_df.groupby(['date'], as_index=False).sum()
spd = slider_df.merge(sentiment_total_pd, left_on = 'date', right_on='date')
spd['sentiment percentage'] = 100*(spd['size_x']/spd['size_y'])

fig2 = func_creating_fig2(spd)

total_number_of_tweets = len(filtered_df['text'])
positive_percentage = 100*len(filtered_df[filtered_df['sentiment']=='Positive'])/len(filtered_df['sentiment'])
neutral_percentage = 100*len(filtered_df[filtered_df['sentiment']=='Neutral'])/len(filtered_df['sentiment'])
negative_percentage = 100*len(filtered_df[filtered_df['sentiment']=='Negative'])/len(filtered_df['sentiment'])

with dataset:
	st.markdown(f"""Displaying tweets from **{date_range[0]}** to **{date_range[1]}**  
	Total Number of Tweets: **{total_number_of_tweets}**  
	Positive: **{positive_percentage:.2f}%**, Neutral: **{neutral_percentage:.2f}%**, Negative: **{negative_percentage:.2f}%**""")

	with col1:
		st.write(fig)
	with col2:
		st.write(fig2)

with common:
	# display tweets
	st.markdown(f"<h5 style='text-align: left;'>Tweets on {game_name}</h5>", unsafe_allow_html=True)
	st.dataframe(filtered_df)

with word_cloud:
	st.write('in progress...')
	# dataset_text = ' '.join(game_dataset['preprocessed tweets'])
	# dataset_text = dataset_text.split()
	# # dataset_text = ''.join(ch for ch in string_value if ch.isalnum())
	# # remove_words = ['https', 'Pokémon', 'pokemon','Pokemon', 'POKEMON','amp','t','co','RT',
	# # 				'X','Y','x','y','Sun','Moon','SunMoon','PokemonSunMoon',
	# # 				'Alpha','Sapphire','Omega', 'Ruby','ORAS',
	# # 				'user']

	# # for word in remove_words:
	# # 	dataset_text = dataset_text.replace(word,'')

	# fig_word, ax = plt.subplots()

	# wordcloud = WordCloud(background_color='white', colormap='Set2',
	# 			collocations=False, stopwords = STOPWORDS).generate(dataset_text)

	# st.caption('Word cloud of most common words between the date range and text search')
	# plt.imshow(wordcloud, interpolation='bilinear')
	# plt.axis("off")
	# plt.show()
	# st.pyplot(fig_word)

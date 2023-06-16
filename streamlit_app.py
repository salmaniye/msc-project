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
import pickle

# set deafult layout to wide
st.set_page_config(layout="wide")
header = st.container()
# initializing containers
with st.sidebar:
	input_container, word_cloud, help_tab = st.tabs(["Control Panel", "WordCloud", "Help"])

st.markdown("""
<style>
.bigger-font {
    font-size:18px !important;
}
</style>
""", unsafe_allow_html=True)

metrics = st.container()
with metrics:
	st.caption('**Please refer to "Help" on the sidebar on how to use the web app**')
	metrics_title = st.container()
	subcol1, subcol2, subcol3 = st.columns(3)
	st.markdown(f'***')

dataset = st.container()
col1, col2 = st.columns(2)
common = st.container()
plot_all = st.container()


# loading metadata
with open(f"metadata/games_metadata.json") as file:
    game_metadata = json.load(file)

# functions
@st.cache_data
def call_dataset(game_name):
	# load data
	game_csv = game_name
	data = pd.read_csv(f'datasets/tweets_datatset_{game_csv}.csv',lineterminator='\n')

	# removing \r in 'sentiment'
	if 'sentiment\r' in data.columns:
		data['sentiment'] = data['sentiment\r'].apply(lambda x: x.replace('\r',''))
		data.drop(columns=['sentiment\r'],inplace=True)

	# changing 'sentiment scores' from str to ndarray
	data['sentiment scores'] = data['sentiment scores'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

	# changing 'created at' date from str to datetime	
	data['datetime'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
	
	# sorting from earliest to latest
	data.sort_values(by='date', inplace=True)
	data.reset_index(drop=True, inplace=True)
	return data

@st.cache_data
def func_sentiment_per_day(df):
	# for creating df with number of tweets
	df = df.groupby(['sentiment','datetime'], as_index=False).size()
	return df

@st.cache_data
def func_slider_df_size(df,date_range):
	# for filtering dates of df with number of tweets
	df = df[df['datetime'].between(date_range[0],date_range[1],inclusive='both')]
	return df

@st.cache_data
def func_slider_df_all(df,date_range):
	# for filtering dates of df with original columns
	df = df[df['datetime'].between(date_range[0],date_range[1],inclusive='both')]
	return df

@st.cache_data
def func_filtered_df(df,options_sentiment):
	# for filtering sentiment
	df = df[df["sentiment"].isin(options_sentiment)]
	return df

@st.cache_data
def func_keyword(df,key):
	# for filtering with text search
	df =  df[df['text'].str.contains(pat=key, case=False)==True]
	return df

with header:
	st.markdown(f"## Sentiment of pre-release tweets on main series Pokémon games")
	st.markdown("""This is a web app that displays tweets and their sentiment on the selected Pokémon game.""")

with open(f"metadata/user_guide.txt") as file:
	user_guide_text = file.read()

# user guide tab
with help_tab:
	st.markdown(user_guide_text)
	st.caption("[*Source*](https://docs.streamlit.io/library/api-reference/data/st.dataframe)")
	st.markdown("***")
	st.markdown("All data, including the app, is stored on [this GitHub repository](https://github.com/salmaniye/msc-project).")
	st.caption('by Salman Fatahillah/sxf181')
	st.markdown("""

		""")

games_list = ['Pokémon X&Y', 'Pokémon Omega Ruby & Alpha Sapphire',
			  'Pokémon Sun & Moon', 'Pokémon Ultra Sun & Ultra Moon',
			  "Pokémon: Let's Go, Pikachu! and Let's Go, Eevee!",
			  'Pokémon Sword & Shield', 'Pokémon Sword and Shield: The Isle of Armor and The Crown Tundra',
			  'Pokémon Brilliant Diamond & Shining Pearl',
			  'Pokémon Legends: Arceus', 'Pokémon Scarlet & Violet']

games_csv = ['01xy','02oras','03sunmoon','04ultrasm','05letsgo','06swsh','07swshdlc','08bdsp','09arceus','10sv']
games_zip = zip(games_list,games_csv)
games_dict = dict(games_zip)
games_dict_index = dict(zip(games_list,range(10)))

# adding about game metadata and metrics
metrics_data = pd.read_csv(f'metadata/games_metrics.csv')

def call_metric_data(game_name):
	index_no = games_dict_index[game_name]
	with metrics_title:
		st.markdown(f'Overall metrics of **{game_name}** (unaffected by options):' if index_no == 0 \
	else f"Overall metrics difference of **{game_name}** from previous game: **{game_metadata[index_no-1]['name']}** (unaffected by options):")
	sentiment_labels = ["Positive", "Neutral", "Negative"]
	for i,column in enumerate([subcol1,subcol2,subcol3]):
		with column:
			delta = None if index_no == 0 else metrics_data.loc[index_no][i+1] - metrics_data.loc[index_no-1][i+1]
			st.metric(label=sentiment_labels[i], value=f'{metrics_data.loc[index_no][i+1]:.2f}%', delta= None if index_no == 0 else f'{delta:.2f}%')

with input_container:
	# a dropdown for the user to choose the game to be displayed
	game_name = st.selectbox('Select a game to display:', games_list)
	st.caption(f'You have selected: {game_name}')

	# About game section
	
	for entry in game_metadata:
		if game_name == entry['name']:
			st.image(entry['image'])
			st.caption(entry['source'])
			with st.expander(f"🎮 About {game_name}"):
				st.markdown(f"""{entry['announced']}  
					{entry['released']}""")
				st.markdown(entry['paragraph1'])
				st.markdown(entry['paragraph2'])
				st.markdown(entry['suggested_searches'])
				call_metric_data(game_name)

@st.cache_data
def func_creating_fig1(df):
	# creates plot of number of tweets and sentiment
	fig = px.line(df, x='datetime', y='size', labels={
		'datetime':'Date',
		'size':'Number of tweets',
		'sentiment':'Sentiment'},
		color='sentiment',
		color_discrete_map={'Positive':'#109618','Neutral':'#3366CC','Negative':'#DC3912'}) 
		#['red', 'blue', 'green']
	fig.update_layout(title_text=f"Number of tweets and their sentiment over time",
		font=dict(size=16))
	return fig

@st.cache_data
def func_creating_fig2(df):
	# creates plot of normalized sentiment with percentage
	fig2 = px.area(df, x='datetime', y='sentiment percentage',labels={
		'datetime':'Date',
		'sentiment percentage':'Sentiment (%)',
		'sentiment':'Sentiment'},
		color='sentiment',
		color_discrete_map={'Positive':'#109618','Neutral':'#3366CC','Negative':'#DC3912'},
		category_orders={"sentiment": ["Negative", "Neutral", "Positive"]})
	fig2.update_layout(title_text=f"Normalized sentiment of tweets over time",
		font=dict(size=16))
	return fig2

game_dataset = call_dataset(games_dict[game_name])


with input_container:
	inputs = st.form(key='form',clear_on_submit=False)

# grouping sentiment per date
sentiment_per_day = func_sentiment_per_day(game_dataset)
min_date = sentiment_per_day['datetime'].min()
max_date = sentiment_per_day['datetime'].max()

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
	date_range[1] = st.date_input('Select end date:', max_date, min_date, max_date,key='dateinput2')

	date_range[0] = pd.to_datetime(date_range[0])
	date_range[1] = pd.to_datetime(date_range[1])

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
	submitted = st.form_submit_button("Click to Submit")

	if submitted:
		st.write("Submitted")
	#create your button to clear the state of the multiselect


with input_container:
	st.button("Reset options to default values", on_click=clear_inputs)
	st.markdown("""

		""")

if keyword_text:
	game_dataset = func_keyword(game_dataset,keyword_text)

# dataframe for number of tweets
sentiment_per_day = func_sentiment_per_day(game_dataset)
slider_df = func_slider_df_size(sentiment_per_day,date_range)
slider_df = slider_df[slider_df["sentiment"].isin(options_sentiment)]

# creates a dataframe of tweets created between dates chosen
date_range_df = func_slider_df_all(game_dataset,date_range)
filtered_df = func_filtered_df(date_range_df,options_sentiment)
if keyword_text:
	filtered_df = func_keyword(filtered_df,keyword_text)

# fig1. sentiment over time
fig = func_creating_fig1(slider_df)

# fig2. normalized sentiment area over time
@st.cache_data
def func_spd(df):
	sentiment_total_pd = df.groupby(['datetime'], as_index=False).sum()
	spd = df.merge(sentiment_total_pd, left_on = 'datetime', right_on='datetime')
	spd['sentiment percentage'] = 100*(spd['size_x']/spd['size_y'])
	return spd

fig2 = func_creating_fig2(func_spd(slider_df))

total_number_of_tweets = len(filtered_df['text'])
positive_percentage = 100*len(filtered_df[filtered_df['sentiment']=='Positive'])/len(filtered_df['sentiment'])
neutral_percentage = 100*len(filtered_df[filtered_df['sentiment']=='Neutral'])/len(filtered_df['sentiment'])
negative_percentage = 100*len(filtered_df[filtered_df['sentiment']=='Negative'])/len(filtered_df['sentiment'])

with dataset:
	st.markdown(f"""<p class="bigger-font">Displaying tweets from <b>{date_range[0].strftime("%Y-%m-%d")}</b>
	to <b>{date_range[1].strftime("%Y-%m-%d")}</b> on <b>{game_name}</b><br>
	Total Number of Tweets: <b>{total_number_of_tweets}</b><br>
	Positive: <b>{positive_percentage:.2f}%</b>,
	Neutral: <b>{neutral_percentage:.2f}%</b>,
	Negative: <b>{negative_percentage:.2f}%</b></p>""",
	unsafe_allow_html=True)

	with col1:
		st.write(fig)
	with col2:
		st.write(fig2)


with common:
	# display tweets
	st.markdown(f"##### Tweets on {game_name}")
	st.dataframe(filtered_df[['text','date','sentiment scores','sentiment']])


with plot_all:
	st.markdown("***")
	fig_all = pickle.load(open('fig_all.pkl','rb'))
	fig_all.update_layout(width=1400)
	st.markdown(f"##### Plot of Number of Tweets from All Games over Time")
	st.plotly_chart(fig_all)

def wordcloud_generator():
	dataset_text = ' '.join(game_dataset['text'])

	with open(f"metadata/custom_stopwords.txt","r") as file:
		custom_stopwords = []
		for line in file:
			line = line.rstrip("\n")
			custom_stopwords.append(line)

	stopwords_all = custom_stopwords + list(STOPWORDS)

	fig_word, ax = plt.subplots()

	wordcloud = WordCloud(background_color='white', colormap='Set2',
				width = 1000, height=2600,
				collocations=False, stopwords = stopwords_all).generate(dataset_text)

	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.show()
	return fig_word

with word_cloud:

	st.markdown('Word cloud of most common words between the date range and text search')
	st.caption('The wordcloud is used to inspire you to find words to use in the text search in the Control Panel')
	fig_word = wordcloud_generator()
	st.pyplot(fig_word)

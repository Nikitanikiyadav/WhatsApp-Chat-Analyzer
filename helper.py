from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

extract = URLExtract()
analyzer = SentimentIntensityAnalyzer()

#fetch stats
def fetch_stats(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]

    words = []
    for message in df['message']:
        words.extend(message.split())

    num_medias = df[df['message'] == '<Media omitted>'].shape[0]

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_medias, len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'user': 'name', 'count': 'percent'})
    return x, df

def create_wordcloud(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=11, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>']

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + " " + str(timeline['year'][i]))
    timeline['time'] = time

    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

def activity_heat_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap


# ---------- SENTIMENT ANALYSIS ----------

def add_sentiment(df):
    df = df.copy()

    def get_sentiment(message):
        if message.strip() in ['<Media omitted>', '']:
            return 'Neutral'
        score = analyzer.polarity_scores(message)['compound']
        if score > 0.05:
            return 'Positive'
        elif score < -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment'] = df['message'].apply(get_sentiment)
    return df

def sentiment_distribution(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    return temp['sentiment'].value_counts()

def sentiment_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']

    timeline = temp.groupby(['year', 'month_num', 'month', 'sentiment']).size().reset_index(name='count')

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + " " + str(timeline['year'][i]))
    timeline['time'] = time

    return timeline

def most_positive_negative_users(df):
    temp = df[df['user'] != 'group_notification']
    positive = temp[temp['sentiment'] == 'Positive']['user'].value_counts().head(5)
    negative = temp[temp['sentiment'] == 'Negative']['user'].value_counts().head(5)
    return positive, negative
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def get_topics(selected_user, df, num_topics=5, num_words=8):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>']

    f = open('stop_hinglish.txt', 'r')
    stop_words = set(f.read().split())

    def clean(message):
        words = [w for w in message.lower().split() if w.isalpha() and w not in stop_words and len(w) > 2]
        return " ".join(words)

    temp = temp.copy()
    temp['clean_message'] = temp['message'].apply(clean)

    # group messages by day into richer "documents" for better topic quality
    documents = temp.groupby('only_date')['clean_message'].apply(lambda x: " ".join(x)).reset_index()
    documents = documents[documents['clean_message'].str.strip() != ""]

    if documents.shape[0] < num_topics + 1:
        return None

    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
    tfidf = vectorizer.fit_transform(documents['clean_message'])

    if tfidf.shape[1] < num_topics:
        return None

    nmf_model = NMF(n_components=num_topics, random_state=42, init='nndsvda', max_iter=500)
    nmf_model.fit(tfidf)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append((f"Topic {topic_idx + 1}", top_features))

    return topics

import networkx as nx

def build_interaction_graph(df, time_gap_minutes=30):
    temp = df[df['user'] != 'group_notification'].copy()
    temp = temp.sort_values('date').reset_index(drop=True)

    edges = {}
    for i in range(1, len(temp)):
        prev_user = temp.loc[i - 1, 'user']
        curr_user = temp.loc[i, 'user']
        prev_time = temp.loc[i - 1, 'date']
        curr_time = temp.loc[i, 'date']

        if prev_user == curr_user:
            continue

        gap = (curr_time - prev_time).total_seconds() / 60
        if gap > time_gap_minutes:
            continue

        pair = (prev_user, curr_user)
        edges[pair] = edges.get(pair, 0) + 1

    G = nx.DiGraph()
    for (u, v), weight in edges.items():
        G.add_edge(u, v, weight=weight)

    return G

def get_most_central_users(G, top_n=5):
    if G.number_of_nodes() == 0:
        return pd.Series(dtype=float)
    centrality = nx.degree_centrality(G)
    series = pd.Series(centrality).sort_values(ascending=False).head(top_n)
    return series
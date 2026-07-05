import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns

st.sidebar.title("WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Select a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    try:
        data = bytes_data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            data = bytes_data.decode("utf-16")
        except UnicodeDecodeError:
            data = bytes_data.decode("utf-8", errors="ignore")

    st.write("Length of data:", len(data))
    st.write("First 300 chars:", data[:300])

    df = preprocessor.preprocess(data)
    df = helper.add_sentiment(df)

    #fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove("group_notification")
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):
        num_messages,words,num_media,links = helper.fetch_stats(selected_user,df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total messages")
            st.title(num_messages)
        with col2:
            st.header("Total words")
            st.title(words)
        with col3:
            st.header("Total Media files shared")
            st.title(num_media)
        with col4:
            st.header("Total links shared")
            st.title(links)
#Topic Modeling
        st.title("Conversation Topics")
        topics = helper.get_topics(selected_user, df)
        if topics is None:
            st.info("Not enough text data to extract topics — try selecting 'Overall' or a chat with more messages.")
        else:
            for topic_name, words in topics:
                st.write(f"**{topic_name}:** " + ", ".join(words))
        #Monthly Timeline
        st.title("Monthly timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax=plt.subplots()

        ax.plot(timeline['time'], timeline['message'],color='Yellow')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily Timeline
        st.title("Daily timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()

        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='Red')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        #Activity map
        st.title("Activity map")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day=helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        #Heatmap
        st.title("Activity HeatMap")
        user_heatmap = helper.activity_heat_map(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        #finding the busiest users in the group(Group level)
        if selected_user=="Overall":
            st.title("Most busy users")
            x,name_with_percent = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values)
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(name_with_percent)

        #creating word cloud
        st.title("WordCloud")
        image = helper.create_wordcloud(selected_user,df)
        fig, ax = plt.subplots()
        ax.imshow(image)
        st.pyplot(fig)

        #Most common words
        most_common_df = helper.most_common_words(selected_user,df)

        fig, ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title("Most common words")
        st.pyplot(fig)

        #Emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            st.pyplot(fig)
            
#Interaction Network
        if selected_user == "Overall":
            st.title("Group Interaction Network")
            G = helper.build_interaction_graph(df)

            if G.number_of_nodes() == 0:
                st.info("Not enough data to build an interaction graph.")
            else:
                fig, ax = plt.subplots(figsize=(8, 8))
                pos = nx.spring_layout(G, seed=42, k=0.8)

                weights = [G[u][v]['weight'] for u, v in G.edges()]
                nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800, ax=ax)
                nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
                nx.draw_networkx_edges(G, pos, width=[w * 0.3 for w in weights], alpha=0.5, arrows=True, ax=ax)

                ax.axis('off')
                st.pyplot(fig)

                st.subheader("Most Central Users")
                central_users = helper.get_most_central_users(G)
                st.dataframe(central_users.rename("Centrality Score"))
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk import word_tokenize, pos_tag
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class Intellibot:
    def __init__(self, call_transcriptions_df):
        self.call_transcriptions_df = call_transcriptions_df

    def welcome_message(self):
        return "Hello, I am Intellibot. What service can I assist you with today?"

    def display_menu(self):
        option = st.selectbox("Select an option", ["Topic Trends", "Agent Performance", "Transfer Calls Analysis", "Sentiment Analysis"])
        return option

    def validate_option(self, option):
        if option not in ["Topic Trends", "Agent Performance", "Transfer Calls Analysis", "Sentiment Analysis"]:
            st.error("Invalid option. Please choose a valid option.")
            return False
        return True
    
    def generate_wordcloud(self,text):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        st.image(wordcloud.to_image())
    
    def extract_nouns(self,text):
        tagged_words = pos_tag(word_tokenize(text))
        nouns = [word for word, pos in tagged_words if pos.startswith('NN')]
        return ' '.join(nouns)

    def trending_topics(self):
        self.call_transcriptions_df
        st.header("Topic Trends")
        
        start_date = st.date_input("Select a start date")
        end_date = st.date_input("Select an end date")
        skill_group_level = st.selectbox("Select a skill group level", self.call_transcriptions_df['skillGroupName'].unique())

        self.call_transcriptions_df['Start_of_Call'] = pd.to_datetime(self.call_transcriptions_df['Start_of_Call'])
        start_date = pd.to_datetime(start_date).tz_localize('UTC')  # Convert start_date to the same datetime format
        end_date = pd.to_datetime(end_date).tz_localize('UTC')
        # Filter the DataFrame based on the selected dates and skill group
        filtered_df = self.call_transcriptions_df[
            (self.call_transcriptions_df['Start_of_Call'] >= start_date) &
            (self.call_transcriptions_df['Start_of_Call'] <= end_date) &
            (self.call_transcriptions_df['skillGroupName'] == skill_group_level)
        ]

        # Topic Distribution - Replaced with Plotly
        st.subheader("Topic Distribution")
        topic_distribution = filtered_df['Primary_Estimated_Topic'].value_counts().reset_index()
        topic_distribution.columns = ['Topic', 'Count']
        # Create a Plotly bar chart
        fig = px.bar(topic_distribution, x='Topic', y='Count',
                    labels={'x': 'Topic', 'y': 'Count'},
                    title="Topic Distribution")
        st.plotly_chart(fig)

        # Pie Chart - Replaced with Plotly
        st.subheader("Pie Chart of Topic Distribution")
        fig = go.Figure(data=[go.Pie(labels=topic_distribution['Topic'], values=topic_distribution['Count'])])
        st.plotly_chart(fig)

        #st.subheader("Word Cloud for Nouns in Topics")
        #topic_text = ' '.join(filtered_df['output_text'])
        #nouns_text = self.extract_nouns(topic_text)
        #self.generate_wordcloud(nouns_text)

        # Topics By Time Of Day
        st.subheader("Topics By Time Of Day")

        # Categorize the time of day
        filtered_df['Time_of_Day'] = pd.cut(
            filtered_df['Start_of_Call'].dt.hour,
            bins=[0, 12, 17, 24],
            labels=['Morning', 'Afternoon', 'Evening']
        )

        # Group by time of day and topic
        time_of_day_data = filtered_df.groupby(['Time_of_Day', 'Primary_Estimated_Topic']).size().unstack(fill_value=0)

        # Check if any time of day doesn't have enough values
        min_values_required = 3  # You can adjust this threshold as needed

        # Create an empty DataFrame to store the sorted data
        time_of_day_data_sorted = pd.DataFrame()

        for time_of_day in time_of_day_data.index:
            if time_of_day_data.loc[time_of_day].sum() < min_values_required:
                st.warning(f"Skipping plotting for {time_of_day} as it doesn't have enough data points ({time_of_day_data.loc[time_of_day].sum()} values).")
                continue

            # Sort the data by time of day in descending order
            time_of_day_data_sorted[time_of_day] = time_of_day_data.loc[[time_of_day]].T.sort_values(time_of_day, ascending=False)[time_of_day]

        # Create a table view
        #st.subheader(f"Topic Distribution by Time of Day")
        #st.dataframe(time_of_day_data_sorted)

        # Filter the table based on time of day
        selected_time_of_day = st.selectbox("Select Time of Day", time_of_day_data.index)

        if len(time_of_day_data_sorted[selected_time_of_day]) < min_values_required:
            st.warning(f"Skipping plotting for {time_of_day} as it doesn't have enough data points")
        else:
            st.subheader(f"Topic Distribution for {selected_time_of_day}")
            selected_time_of_day_data = time_of_day_data_sorted[selected_time_of_day]
            st.dataframe(selected_time_of_day_data)



        # Topic, Agent Based Sentiment Score of Conversations (you need to implement this)
        st.subheader("Topic, Agent Based Sentiment Score of Conversations")
        # Implement the sentiment analysis and visualization here


        # Number of summaries to display
        num_summaries = st.selectbox("Select the number of summaries to display", [5, 10, 15])

        # Get the most recent summaries, primary topics, and raw transcriptions
        recent_data = filtered_df[['call_sid', 'clean_summary', 'Primary_Estimated_Topic', 'output_text']].tail(num_summaries)


        # Create a user-friendly visual representation of summaries and topics with raw transcriptions
        st.subheader("Visual Representation")

        # Loop through the recent data and display each summary, its primary topic, and add a dropdown for raw transcription
        for index, row in recent_data.iterrows():
            with st.expander(f"Summary for Call SID: {row['call_sid']}"):
                st.info(f"Primary Topic: {row['Primary_Estimated_Topic']}")
                st.write(f"Summary: {row['clean_summary']}")
                st.write('---')  # Add a horizontal line to separate entries

            with st.beta_expander(f"View Raw Transcription for Call SID: {row['call_sid']}"):
                st.write(row['output_text'])



    def run(self):
        st.title("Intellibot Chat")
        st.write(self.welcome_message())

        option = self.display_menu()

        if self.validate_option(option):
            if option == "Topic Trends":
                self.trending_topics()
            elif option == "Agent Performance":
                self.agent_performance()
            elif option == "Transfer Calls Analysis":
                self.transfer_calls_analysis()
            elif option == "Sentiment Analysis":
                self.sentiment_analysis()

if __name__ == '__main__':
    # Load your data here
    call_transcriptions_df = pd.read_csv(r"C:\Users\307164\Desktop\2023_CS_PA_NLP_Final\cs_pa_nlp\Streamlit_Application\final_output_for_streamlit.csv")
    bot = Intellibot(call_transcriptions_df)
    bot.run()
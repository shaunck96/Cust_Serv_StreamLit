import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk import word_tokenize, pos_tag
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Change the Streamlit theme and layout
st.set_page_config(
    page_title="Intellibot Chat",
    page_icon=":robot_face:",
    layout="wide",
)

# Custom CSS for the app
st.markdown("""
    <style>
    body {
        background-image: url('./bg_image.jpg');
        background-size: cover;
    }
    .custom-header {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
    }
    .custom-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

class Intellibot:
    def __init__(self, call_transcriptions_df):
        self.call_transcriptions_df = call_transcriptions_df

    def welcome_message(self):
        st.markdown("<p class='custom-header'>Hello, I am Intellibot. What service can I assist you with today?</p>", unsafe_allow_html=True)

    def display_menu(self):
        option = st.selectbox("Select an option", ["Topic Trends", "Agent Performance", "Transfer Calls Analysis", "Sentiment Analysis"])
        return option

    def validate_option(self, option):
        if option not in ["Topic Trends", "Agent Performance", "Transfer Calls Analysis", "Sentiment Analysis"]:
            st.error("Invalid option. Please choose a valid option.")
            return False
        return True
    
    def generate_wordcloud(self, text):
        wordcloud = WordCloud(width=800, 
                              height=400, 
                              background_color='white').generate(text)
        st.image(wordcloud.to_image())
    
    def extract_nouns(self, text):
        tagged_words = pos_tag(word_tokenize(text))
        nouns = [word for word, pos in tagged_words if pos.startswith('NN')]
        return ' '.join(nouns)

    def trending_topics(self):
        self.call_transcriptions_df
        st.header("Topic Trends")
        
        start_date = st.date_input("Select a start date")
        end_date = st.date_input("Select an end date")
        skill_group_level = st.selectbox("Select a skill group level", self.call_transcriptions_df['skillGroupName'].unique())

        self.call_transcriptions_df['Start_of_Call'] = pd.to_datetime(
            self.call_transcriptions_df['Start_of_Call'])
        start_date = pd.to_datetime(
            start_date).tz_localize('UTC')  # Convert start_date to the same datetime format
        end_date = pd.to_datetime(end_date).tz_localize('UTC')
        # Filter the DataFrame based on the selected dates and skill group
        filtered_df = self.call_transcriptions_df[
            (self.call_transcriptions_df['Start_of_Call'] >= start_date) &
            (self.call_transcriptions_df['Start_of_Call'] <= end_date) &
            (self.call_transcriptions_df['skillGroupName'] == skill_group_level)
        ]

        # Topic Distribution - Replaced with Plotly
        topic_distribution = filtered_df['Primary_Estimated_Topic'].value_counts().reset_index()
        topic_distribution.columns = ['Topic', 'Count']
        topic_distribution = topic_distribution.sort_values(by=['Count'],ascending=False).head(10)
        # Create a Plotly bar chart
        #fig = px.bar(topic_distribution, x='Topic', y='Count',
        #            labels={'x': 'Topic', 'y': 'Count'},
        #            title="Topic Distribution")
        #st.plotly_chart(fig)

        # Pie Chart - Replaced with Plotly
        #st.subheader("Pie Chart of Topic Distribution")
        #fig = go.Figure(data=[go.Pie(labels=topic_distribution['Topic'], values=topic_distribution['Count'])])
        #st.plotly_chart(fig)

        # Create an interactive pie chart
        fig = px.pie(topic_distribution, values='Count', names='Topic', 
                    title='Topic Distribution', 
                    hole=0.3, # Make it a donut chart (0 for a pie chart)
                    labels={'Topic': 'Topics'}, 
                    color_discrete_sequence=px.colors.qualitative.Set3) # Set color scheme

        fig.update_traces(textinfo='percent+label', pull=[0.2, 0.2, 0.2, 0.2]) # Customize labels and pull slices

        # Custom layout
        fig.update_layout(
            showlegend=False,
            legend=dict(title='Topics', x=2, y=0.5)
        )

        # Show the chart using Streamlit
        st.plotly_chart(fig, use_container_width=True)

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

        # Filter the table based on time of day
        selected_time_of_day = st.selectbox("Select Time of Day", time_of_day_data.index, key="time_of_day_selectbox")

        filtered_df = filtered_df[['call_sid', 'clean_summary', 'Primary_Estimated_Topic', 
                                   'Secondary_Estimated_Topic',
                                   "skillGroupName",
                                    "workerName",
                                    "workerManager",
                                    "talkTime",
                                    "holdTime",
                                    "acwTime",
                                    "User_Response",
                                    "Intent",
                                    "Intent_Score",
                                   'output_text', 'Time_of_Day']]
        filtered_recent_data = filtered_df[filtered_df['Time_of_Day']==selected_time_of_day]

        if len(filtered_recent_data) < min_values_required:
            st.warning(f"Skipping plotting for {selected_time_of_day} as it doesn't have enough data points")

        else:
            #st.subheader(f"Topic Distribution for {selected_time_of_day}")
            #st.dataframe(filtered_recent_data.groupby(['Primary_Estimated_Topic'])['call_sid'].count().sort_values(ascending=False))
            
            #num_summaries = st.selectbox("Select the number of summaries to display", [5, 10, 15], key="num_summaries_selectbox")
            #filtered_recent_data = filtered_recent_data.head(num_summaries)
            
            #st.subheader("Summaries With Original Transcriptions")

            
            #for index, row in filtered_recent_data.iterrows():
            #    with st.expander(f"Summary for Call SID: {row['call_sid']}"):
            #        st.subheader("Call Details")
            #        st.info(f"Primary Topic: {row['Primary_Estimated_Topic']}")
            #        st.info(f"Secondary Topic: {row['Secondary_Estimated_Topic']}")
            #        st.info(f"Destination Skill Group: {row['skillGroupName']}")
            #        st.info(f"Agent: {row['workerName']}")
            #        st.info(f"Manager: {row['workerManager']}")

            #        st.subheader("Call Durations")
            #        st.info(f"Talk Time: {row['talkTime']}")
            #        st.info(f"Hold Time: {row['holdTime']}")
            #        st.info(f"After Call Work Time: {row['acwTime']}")

            #        st.subheader("User IVR Interaction")
            #        st.info(f"User Utterance To IVR: {row['User_Response']}")
            #        st.info(f"Intent Identified (IVR): {row['Intent']}")
            #        st.info(f"Intent Identification Confidence Score (IVR): {row['Intent_Score']}")
            #        st.write(f"Summary: {row['clean_summary']}")
            #        st.markdown('---')  # Add a horizontal line to separate entries

            #    with st.expander(f"View Raw Transcription for Call SID: {row['call_sid']}"):
            #        st.write(row['output_text'])


            # Display topic distribution
            st.subheader(f"Topic Distribution for {selected_time_of_day}")
            topic_distribution = filtered_recent_data.groupby(['Primary_Estimated_Topic'])['call_sid'].count().sort_values(ascending=False)
            st.dataframe(topic_distribution)

            # Select the number of summaries to display
            num_summaries = st.selectbox("Select the number of summaries to display", [5, 10, 15], key="num_summaries_selectbox")

            # Get the most recent summaries, primary topics, and raw transcriptions
            filtered_recent_data = filtered_recent_data.sample(n=num_summaries, random_state=42)

            # Create a user-friendly visual representation of summaries and topics with raw transcriptions
            st.subheader("Summaries With Original Transcriptions")

            # Loop through the filtered recent data and display each summary and its primary topic
            for index, row in filtered_recent_data.iterrows():
                with st.expander(f"Summary for Call SID: {row['call_sid']}"):
                    # Call details section
                    st.subheader("Call Details")
                    st.write(f"Primary Topic: {row['Primary_Estimated_Topic']}")
                    st.write(f"Secondary Topic: {row['Secondary_Estimated_Topic']}")
                    st.write(f"Destination Skill Group: {row['skillGroupName']}")
                    st.write(f"Agent: {row['workerName']}")
                    st.write(f"Manager: {row['workerManager']}")

                    # Call durations section
                    st.subheader("Call Durations")
                    st.write(f"Talk Time: {row['talkTime']}")
                    st.write(f"Hold Time: {row['holdTime']}")
                    st.write(f"After Call Work Time: {row['acwTime']}")

                    # User IVR Interaction section
                    st.subheader("User IVR Interaction")
                    st.write(f"User Utterance To IVR: {row['User_Response']}")
                    st.write(f"Intent Identified (IVR): {row['Intent']}")
                    st.write(f"Intent Identification Confidence Score (IVR): {row['Intent_Score']}")
                    st.write(f"Summary: {row['clean_summary']}")

                    # Horizontal line to separate entries
                    st.markdown('---')

                with st.expander(f"View Raw Transcription for Call SID: {row['call_sid']}"):
                    st.code(row['output_text'])  # Display raw transcription as code for better readability

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
    call_transcriptions_df = pd.read_csv(r"./final_output_for_streamlit_v2.csv")
    bot = Intellibot(call_transcriptions_df)
    bot.run()

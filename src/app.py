import os
from flask import Flask, request, jsonify
from langchain import OpenAI
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
import pathway as pw
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Fetching the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the Flask app
app = Flask(__name__)

# Setup the OpenAI model using Langchain, with the API key
llm = OpenAI(api_key=OPENAI_API_KEY)

# Class to simulate fetching event details using Pathway
class EventDataStream:
    def __init__(self):
        # Example of event data; in real-world, you could use something like Eventbrite API
        self.event_df = pd.DataFrame({
            'event': ['Concert', 'Art Exhibition', 'Film Screening', 'Local Food Fest'],
            'location': ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad'],
            'date': ['2024-09-20', '2024-09-21', '2024-09-22', '2024-09-23']
        })
    
    def fetch_events(self):
        # In real life, this will call an API to get event data
        return self.event_df

# Function to handle user profile data
def process_profile_data(user_data):
    # Here, you can include things like common interests, shared PDFs, etc.
    return f"User interests: {user_data['interests']}."

# Function to create an index from PDF files using LlamaIndex
def create_pdf_index(pdf_folder_path):
    # Loading documents from the given folder and creating an index from them
    documents = SimpleDirectoryReader(pdf_folder_path).load_data()
    index = GPTSimpleVectorIndex.from_documents(documents)
    return index

# Function to generate conversation ideas based on user interests and event data
def generate_conversation(user_data):
    # Fetching event data (this could be real-time if connected to a live API)
    events = EventDataStream().fetch_events()
    
    # Process the user's profile data (like hobbies, favorite topics, etc.)
    profile_info = process_profile_data(user_data)
    
    # Taking 2 upcoming events as suggestions
    event_suggestions = events.head(2).to_dict(orient="records")
    
    # Building a prompt for the OpenAI LLM model
    prompt = (
        f"Suggest a conversation starter for a user with {profile_info}. "
        f"Upcoming events are: {event_suggestions}."
    )
    
    # Get the LLM response based on the prompt
    response = llm(prompt)
    return response

# Route to handle POST requests for generating conversation starters
@app.route('/generate', methods=['POST'])
def generate_suggestions():
    # Extracting user data from the request body (assuming it's in JSON format)
    user_data = request.json
    
    # Generating conversation ideas based on the profile and event data
    conversation_starter = generate_conversation(user_data)
    
    # Returning the generated conversation starter in a JSON response
    return jsonify({"conversation_starter": conversation_starter})

# Main entry point to run the Flask app
if __name__ == '__main__':
    # Running the Flask app, accessible on 0.0.0.0 at port 5000
    app.run(host='0.0.0.0', port=5000)

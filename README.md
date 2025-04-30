# Romeo-AI-Chatbot-with-Memory
Romeo is an AI chatbot built using DialoGPT. Created by Nishanth, an AI &amp; DS student from Rathinam College, it offers personalized conversations, remembers chat history, and provides responses like time and creator details. It’s easily customizable for interactive chats.
pip install transformers
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the pre-trained DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize chat history
chat_history_ids = None

# Function to get current time
def get_time():
    return datetime.datetime.now().strftime("%H:%M:%S")

# Function to handle conversation with Romeo
def chat_with_romeo(input_text):
    global chat_history_ids

    # Special case responses for questions about creator
    if "who created you" in input_text.lower():
        return "I was created by Nishanth, a student of AI & DS from Rathinam College."

    # Special case responses for questions about identity
    if "who are you" in input_text.lower():
        return "I am Romeo, your friendly AI chatbot!"

    if "what is your name" in input_text.lower():
        return "My name is Romeo. Nice to meet you!"

    # Special case for time-related questions
    if "what is the time" in input_text.lower():
        return f"The current time is {get_time()}."

    # Encode the new user input and add the end-of-sentence token
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

    # Concatenate the new input with the previous chat history
    bot_input_ids = new_user_input_ids
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

    # Generate a response from the model
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000, 
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3, 
        top_p=0.9, 
        temperature=0.7, 
        do_sample=True  # Set to True to enable sampling
    )

    # Decode the generated response and return it
    bot_output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return bot_output

# Start the conversation loop
print("🤖 Welcome to Romeo! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Goodbye! :)")
        break
    response = chat_with_romeo(user_input)
    print("Romeo:", response)

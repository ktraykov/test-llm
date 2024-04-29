#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
simplechat.py

Simple Chat application

Author: Kiril Traykov
Email: kiril.traykov@gmail.com
Date: 28.04.2024

License: MIT License

Description:
This is a simple command line application developed as part of the article "A Framework for Security Testing of Large Language Model 
Proceedings of 2024 IEEE 12th International Conference on Intelligent Systems (IS)

Usage:
Run the simplechat.py file and submit your query to the LLM and see the response it generates.

Notes:
The app is using the lasets open source LLMs available in Groq LPU Inference Engine. 
You need to provide your own API key to connect and run inference. 
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder)
from langchain.memory import ConversationBufferWindowMemory

# Define the language model (groq)
#available models
#llama3-70b-8192
#llama3-8b-8192
#mixtral-8x7b-32768
#gemma-7b-it

#set API key
GROQ_KEY = "gsk_YOUR_OWN_API_KEY"

llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key = GROQ_KEY)

#define memory window

memory = ConversationBufferWindowMemory(memory_key="chat_history",
                                        input_key='input', 
                                        k = 5,
                                        return_messages=True, 
                                        output_key='text')
#set prompt template
prompt = ChatPromptTemplate.from_messages(
                  [
                    # SystemMessage(
                    #     content=PROMPT_MAI
                    # ),
                    MessagesPlaceholder(variable_name="chat_history"),  # Where the memory will be stored.
                    HumanMessagePromptTemplate.from_template("{input}"),  # Where the human input will injected
                ]
            )

chatbot = LLMChain(
                    llm=llm, 
                    prompt=prompt, 
                    verbose=False, 
                    memory=memory,
                    )            

#fuction to be tested
def handle_message(chat,message):
  """
  Processes the user message and returs a response using the language model.
  """
  # Generate response using the language model
  response = chat.invoke(message)['text']
  return response


#main loop
def main():
    while True:
        # Get user input
        user_message = input("Your message:> ")
        
        if user_message == "quit":
            break
        
        # Handle the message and generate a response
        output = handle_message(chatbot,user_message)
        
        # Print the bot's response 
        print(f"Chatbot reponse: {output}")

if __name__ == "__main__":
    main()

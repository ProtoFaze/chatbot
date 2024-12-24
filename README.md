# 💬 RAG chatbot

A simple streamlit app that does RAG 

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-template.streamlit.app/)

### How to run it on your own machine
1. setup your environment secrets, refer to .streamlit/secrets.toml.example

2. Install the requirements

   ```
   $ cd src
   $ pip install -r requirements.txt
   ```

3. Run the app

   ```
   $ streamlit run Chat.py
   ```

### How to setup your own remote llm endpoint with ollama and modal labs
> **Disclaimer:** This backend server is strictly for demonstrating a proof-of-concept for running LLMs on modal labs via ollama, you are encouraged to support Modal labs via their payed services for a better experience, support, and system scalability

1. Setup an account on modal labs  
   [modal labs sign-up page](https://modal.com/signup?next=%2Fapps)

2. Pull this repo and redirect to the project directory using the terminal// command line interface
   ```
   $ git clone https://github.com/ProtoFaze/chatbot.git   
   $ cd chatbot
   ```

3. Install modal lab's python library using pip.
   ```
   $ pip install modal
   ```

4. Follow the instructions in your browser to authenticate your computer for controlling your modal account
   ```
   $ modal setup
   ```

5. Change your directory again to the backend codebase and deploy the server
   ```
   $ cd ollama_backend
   $ modal deploy ollama_backend.py
   ```
   Your terminal / command line interface should show a url to access the server

6. In your frontend, navigate to the settings page, and fill in the previous url into the ollama endpoint field and submit.

7. Try running your own server in the chat page now.

# Salesman Chatbot
This project is a step by step development project of an insurance salesman chatbot,    
The chatbot should be able to access the corpus of data related to a product it is advertising,   
The bot POC is set up using open source and free tier options, no free trials from providers required.


# Tech/AI stack  
POLM  
Python version 3.12  
Ollama for edge device language model hosting  
LlamaIndex for DAQ and indexing  
Modal labs for hosting asgi web endpoints and llm inference  
MongoDB Atlas for data storage  

# concepts
RAG  
Structured LLM output  

# Further Improvements
- [x] pdf processing pipeline  (via notebook)
- [x] structured data corpus fetch
- [x] basic chatlog analysis

# Acknowledgement
Thanks for sharing these demos and blogs  
using streamlit with ollama for prototyping   
[demo of streamlit with ollama](https://github.com/tonykipkemboi/ollama_streamlit_demos/blob/main/01_%F0%9F%92%AC_Chat.py)  

using llamaparse with ollama  
[the repo](https://github.com/sudarshan-koirala/llamaparser-example/blob/main/parser-ollama.py)   
[the blog article](https://medium.com/@sudarshan-koirala/super-easy-way-to-parse-pdfs-a528fc9c2ea6)

using ollama as a freemium backend service  
[run ollama with modal](https://github.com/irfansharif/ollama-modal)

setting up response streaming via fastAPI(compatible with modal labs)  
[FastAPI Streaming Response: Error: Did not receive done or success response in stream](https://kontext.tech/article/1377/fastapi-streaming-response-error-did-not-receive-done-or-success-response-in-stream) 

using structured outputs on ollama  
[Structured outputs](https://ollama.com/blog/structured-outputs)

sharing how to control page visibility on streamlit  
[Hide/show pages in multipage app based on conditions](https://discuss.streamlit.io/t/hide-show-pages-in-multipage-app-based-on-conditions/28642)
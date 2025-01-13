# 💬 RAG chatbot

A simple streamlit app that does RAG 

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://poc-gmbis-chatbot.streamlit.app/)

### How to run it on your own machine
1. Setup your environment secrets, refer to .streamlit/secrets.toml.example

2. Install the dependencies

   ```
   $ cd src
   $ pip install -r src/requirements.txt
   ```

3. Run the app

   ```
   $ streamlit run src/Chat.py
   ```

### How to setup your own remote llm endpoint with ollama and modal labs
> **Disclaimer:** This backend server is strictly for demonstrating a proof-of-concept for running LLMs on modal labs via ollama, you are encouraged to support Modal labs via their payed services for a better experience, support, and system scalability, do check out their [pricing page](https://modal.com/pricing) if interested

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

4. Follow the instructions in your browser to authenticate your device for accessing your modal account
   ```
   $ modal setup
   ```

5. Change your directory again to the backend codebase and deploy the server
   ```
   $ cd ollama_backend
   $ modal deploy ollama_backend.py
   ```
   Your terminal / command line interface should show a url to access the server

6. Change the endpoint
   - In your frontend, navigate to the settings page, and fill in the previous url into the ollama endpoint field and submit, 
   - or just change the environment variable or secrets that your frontend accesses for ollama endpoints.  
   (uses `.toml` by default, feel free to refactor to use `.env`)

7. Try running your own server in the chat page now.

# Salesman Chatbot
This is a step by step development of an insurance salesman chatbot proof-of-concept,    
The chatbot should be able to access the corpus of data related to a product it is advertising,   
The POC is set up using open source and free tier options, no free trials from providers required.


# Tech/AI stack  
POLM  
Python version 3.12  
Ollama for edge device language model hosting  
LlamaIndex for parsing and ingestion
Modal labs for provisioning computes to develop and test with ASGI web endpoints and llm inference  
MongoDB Atlas for data storage  
Streamlit for user interface  
# concepts
Retrieval Augmented Generation (RAG)  
Structured LLM output  
Few-shot prompting

# Further Improvements
- [x] pdf processing workflow  (via notebook)
- [x] structured data corpus fetch
- [x] structured outputs via json schema
- [x] basic chatlog analysis and control
- [x] sample questions integration via database connection
- [x] chatbot abuse siteblock
- [x] timeout message

# Acknowledgement
Thanks for sharing these demos and blogs:  

Using streamlit with Ollama for prototyping   
[demo of streamlit with ollama](https://github.com/tonykipkemboi/ollama_streamlit_demos/blob/main/01_%F0%9F%92%AC_Chat.py)  

Using llamaparse with Ollama  
[the repo](https://github.com/sudarshan-koirala/llamaparser-example/blob/main/parser-ollama.py)   
[the blog article](https://medium.com/@sudarshan-koirala/super-easy-way-to-parse-pdfs-a528fc9c2ea6)  

Integrating Llamaparse vector indexes with Mongodb  
[How to Build a RAG System With LlamaIndex, OpenAI, and MongoDB Vector Database](https://www.mongodb.com/developer/products/atlas/rag-with-polm-stack-llamaindex-openai-mongodb/)  

Using Ollama as a freemium backend service  
[run ollama with modal](https://github.com/irfansharif/ollama-modal)

Setting up response streaming via fastAPI(compatible with modal labs)  
[FastAPI Streaming Response: Error: Did not receive done or success response in stream](https://kontext.tech/article/1377/fastapi-streaming-response-error-did-not-receive-done-or-success-response-in-stream) 

Using structured outputs on Ollama  
[Structured outputs](https://ollama.com/blog/structured-outputs)

Controling page visibility on Streamlit  
[Hide/show pages in multipage app based on conditions](https://discuss.streamlit.io/t/hide-show-pages-in-multipage-app-based-on-conditions/28642)
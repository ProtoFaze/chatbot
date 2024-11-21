FROM python:3.12.7-slim
WORKDIR /src
ENV MONGODB_URI = 'mongodb+srv://damonngkhaiweng:jhT8LGE0qi6XsKfz@chatbotcluster.noyfa.mongodb.net/?retryWrites=true&w=majority&appName=chatbotcluster'
ENV PINECONE_API_KEY = 'ce2e7e04-18d6-4408-a9ab-7527162af1d7'
ENV MONGODB_DB = 'product1'
ENV PINECONE_INDEX_NAME = 'product1'
ENV STREAMLIT_SERVER_PORT = 8501
ENV STREAMLIT_APP_FILE = 'src/app.py'
COPY requirements.txt requirements.txt
RUN pip install -r src/requirements.txt
EXPOSE 8501
COPY . .
CMD ['streamlit', 'run', $STREAMLIT_APP_FILE]

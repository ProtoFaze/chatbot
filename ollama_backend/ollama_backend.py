import json
import requests
import ollama
import modal
import subprocess
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, Request
from pydantic  import BaseModel
from typing import Literal

class IntentModel(BaseModel):
    reasoning: str
    intent: Literal["normal","register","rag","verify"]

app = FastAPI()

# Setup the model store volume to avoid repeated downloads.
volume = modal.Volume.from_name("ollama-store", create_if_missing=True)
model_store_path = "/vol/models"
#mount the local directory storing our ollama model filesto the remote volume
mount = modal.Mount.from_local_dir(local_path='modelfiles', remote_path="modelfiles")

def serve_ollama():
    '''Ensure Ollama server is running.'''
    subprocess.Popen(["ollama", "serve"])

def ollama_version():
    '''Get the version of the Ollama server.'''
    res = subprocess.run(["ollama", "--version"])
    # print(res)
    return str(res)

image = (modal.Image
        .debian_slim()
        .apt_install("curl")
        .run_commands("curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.5.4 sh")
        .run_commands("apt remove -y curl")
        .pip_install("ollama")
        .env({'OLLAMA_MODELS': model_store_path})
        .pip_install("fastapi[standard]","pydantic",'requests')
        .run_function(serve_ollama)
)

web_app = FastAPI()
app = modal.App(name="llama_ptfz")

@app.cls(
    gpu='T4',
    allow_concurrent_inputs=10,
    volumes={model_store_path: volume},
    image=image,
    mounts=[mount],
    container_idle_timeout=60,
)
class Ollama:
    '''Ollama class for handling calls to the endpoint'''
    BASE_URL: str = "http://localhost:11434"
    
    @modal.enter()
    def init_model(self):
        '''Start the Ollama server.'''
        print("Starting server")
        serve_ollama()

    @modal.method()
    def version(self):
        '''Print the version of the Ollama server.'''
        return ollama_version()
    
    @modal.method()
    def warmup(self, model: str, **kwargs):
        '''Warmup the model.'''
        print(f'loading {model}')
        try: #warmup language models with generate
            ping = ollama.generate(model=model, **kwargs)
        except ollama._types.ResponseError as e: #fallback for embedding models
            ping = ollama.embed(model=model, input="", **kwargs)
        return f"{model} warmed up"

    @modal.method()
    def chat(self, model: str, messages: list, tools: list = [], stream=True, **kwargs):
        '''Handle chat interaction and stream the response.'''
        payload = {"model":model,"messages":messages,"tools":tools,"stream":stream, **kwargs}

        # if 'format' in payload and not isinstance(payload['format'], dict):
        #     print("formated")
        #     payload['format'] = json.dumps(payload['format'])
        print(f'''payload:
              {payload}''')
        response = requests.post(self.BASE_URL+"/api/chat", json=payload, stream=stream)
        if stream:
            print("Streaming")
            for chunk in response.iter_lines():
                yield chunk+b"\n"
        elif response is not None:
            print("Not streaming")
            print(response.content)
            yield json.loads(response.content)
    
    @modal.method()
    def generate(self, model: str, prompt: str, stream=True, **kwargs):
        '''Generate a response from the given user prompt.'''
        payload = {"model":model,"prompt":prompt,"stream":stream, **kwargs}
        response = requests.post(self.BASE_URL+"/api/generate", json=payload, stream=stream)
        if stream:
            print("Streaming")
            for chunk in response.iter_lines():
                yield chunk+b"\n"
        elif response is not None:
            print("Not streaming")
            yield json.loads(response.content)

    @modal.method()
    def embed(self, model: str, input: str, **kwargs):
        '''Embed a given text.'''
        payload = {"model":model,"input":input, **kwargs}
        response = requests.post(self.BASE_URL+"/api/embed", json=payload)
        embeddings = json.loads(response.content).get("embeddings")
        print("returned embeddings")
        print(embeddings)
        # print(len(embeddings[0]))
        return json.loads(response.content)

    @modal.method()
    def embeddings(self, model: str, prompt: str, **kwargs):
        '''Embed a given text.'''
        payload = {"model":model,"prompt":prompt, **kwargs}
        print("payload \\|/")
        print(payload)
        response = requests.post(self.BASE_URL+"/api/embeddings", json=payload)
        embeddings = json.loads(response.content)
        print("returned embeddings")
        print(embeddings)
        print(len(embeddings))
        return json.loads(response.content)

    @modal.method()
    def list(self):
        models = requests.get(self.BASE_URL+"/api/tags")
        print(models.content)
        return json.loads(models.content) if models else b'[]'
    
    @modal.method()
    def list_running(self):
        '''List all running models.'''
        models = requests.get(self.BASE_URL+"/api/ps")
        print(models.content)
        return json.loads(models.content) if models else b'[]'

    @modal.method()
    def pull(self, model: str, stream = True, **kwargs):
        '''Pull a model from the model store.'''
        payload = {"model":model, "stream":stream, **kwargs}
        response = requests.post(self.BASE_URL+"/api/pull", json=payload, stream=stream)
        if stream:
            for chunk in response.iter_lines():
                yield chunk+b"\n"
        elif response is not None:
            yield json.loads(response.content)

    @modal.method()
    def create(self, model:str, modelfile: str = None, path: str = None, stream = True, **kwargs):
        '''Create a new model.'''
        payload = {"model":model, "modelfile":modelfile, "path":path, "stream":stream, **kwargs}
        response = requests.post(self.BASE_URL+"/api/create", json=payload, stream=stream)
        if stream:
            for chunk in response.iter_lines():
                yield chunk+b"\n"
        elif response is not None:
            yield json.loads(response.content)

@web_app.post("/api/warmup")
async def warmup(request: Request):
    '''Warmup the model'''
    ollama = Ollama()
    params = await request.json()
    res = ollama.warmup.remote(**params)
    return JSONResponse(content=res)

@web_app.post("/api/chat")
async def chat(request: Request):
    '''abstraction layer to receive and redirect the request to an instantiated ollama client's chat completion endpoint'''
    ollama = Ollama()
    params = await request.json()
    stream = params.get("stream", True)
    if stream:
        print("using streamRes")
        return StreamingResponse(ollama.chat.remote_gen(**params), media_type="application/x-ndjson")
    else:
        print("using JSONRes")
        response = list(ollama.chat.remote_gen(**params))[0]
        print(type(response))
        print(f"after dict conversion {response}")
        return JSONResponse(content=response)

@web_app.post("/api/generate")
async def generate(request: Request):
    '''abstraction layer to receive and redirect the request to an instantiated ollama client's normal completion endpoint'''
    ollama = Ollama()
    params = await request.json()
    stream = params.get("stream", True)
    if stream:
        print("using streamRes")
        return StreamingResponse(ollama.generate.remote_gen(**params), media_type="application/x-ndjson")
    else:
        print("using JSONRes")
        response = list(ollama.generate.remote_gen(**params))[0]
        print(type(response))
        print(f"after dict conversion {response}")
        return JSONResponse(content=response)

@web_app.post("/api/embed")
async def embed(request: Request):
    '''Get vector embeddings for given text'''
    print("vectorizing text")
    params = await request.json()
    res = Ollama().embed.remote(**params)
    return JSONResponse(content=res)

@web_app.post("/api/embeddings")
async def embed(request: Request):
    '''
    DEPRECATED--Get vector embeddings for given text, use /api/embed instead,
    only maintained for backwards compatibility with llamaParse
    '''
    print("vectorizing text")
    params = await request.json()
    res = Ollama().embeddings.remote(**params)
    return JSONResponse(content=res)

@web_app.get("/api/tags")
async def tags():
    '''Get list of models'''
    print("Getting list of models")
    res = Ollama().list.remote()
    return JSONResponse(content=res)

@web_app.get("/api/ps")
async def ps():
    '''Get list of running models'''
    print("Getting list of models")
    res = Ollama().list_running.remote()
    return JSONResponse(content=res)

@web_app.post("/api/pull")
async def pull(request: Request):
    '''Pull a model from the model store'''
    print("Pulling model")
    ollama = Ollama()
    params = await request.json()
    stream = params.get("stream", True)
    if stream:
        print("using streamRes")
        return StreamingResponse(ollama.pull.remote_gen(**params), media_type="application/x-ndjson")
    else:
        print("using JSONRes")
        response = list(ollama.pull.remote_gen(**params))[0]
        print(type(response))
        print(f"after dict conversion {response}")
        return JSONResponse(content=response)

@web_app.post("/api/create")
async def create(request: Request):
    '''
    Create a new model, based on parameters within a json request, such as
    model name, path to model files, and whether to stream the progress response
    '''
    print("Creating model")
    ollama = Ollama()
    params = await request.json()
    stream = params.get("stream", True)
    if stream:
        print("using streamRes")
        return StreamingResponse(ollama.create.remote_gen(**params), media_type="application/x-ndjson")
    else:
        print("using JSONRes")
        response = list(ollama.create.remote_gen(**params))[0]
        print(type(response))
        print(f"after dict conversion {response}")
        return JSONResponse(content=response)

@web_app.get("/version")
async def version():
    '''Get the version of the Ollama server'''
    print(Ollama().version.remote())
    res = Ollama().version.remote()
    return JSONResponse(content=res)

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app


@app.local_entrypoint()
def init_and_setup():
    #setup api
    ollama_client = ollama.Client(host="https://protofaze--llama-ptfz-fastapi-app-dev.modal.run")

    print("\npulling models")
    originalModels = ['llama3.2', 'nomic-embed-text']
    for model in originalModels:
        for progress in ollama_client.pull(model=model, stream=True):
            print(progress)
    print("\ncreating models")
    models = ["C3","intentClassifier"]
    for model in models:
        for progress in ollama_client.create(model=model, path='modelfiles/'+model,stream=True):
            print(progress.status)

    #setup test variables
    prompt = 'hi'
    messages = [{'role':'user','content':prompt}]
    multiturn_messages = messages+[
                          {"role": "assistant", "content": "Hi, my name is C3 your dedicated assistant for any enquiry about the group multiple benefit insurance plan by great eastern. How can I help you today?"},
                          {"role": "user", "content": "what can you tell me about the plan"}
                        ]
    intentClassifier = 'intentClassifier'
    llm = 'llama3.2'
    
    print("\ntest for non streaming chat completion (structured response)")
    res = ollama_client.chat(model=intentClassifier,
                             messages=multiturn_messages,
                             stream=False,
                             format=IntentModel.model_json_schema()
                             )
    response = IntentModel.model_validate_json(res.message.content)
    print(response)

    print("\ntest for streaming chat completion")
    res = ollama_client.chat(messages=messages, model=llm, stream=True)
    for chunk in res:
        print(chunk.get('message').get("content"), end='', flush=True)


    print("\ntest for non streaming normal completion")
    res = ollama_client.generate(model=llm, prompt=prompt, stream=False)
    print(res.get('response'))
    print("\ntest for streaming normal completion")
    res = ollama_client.generate(model=llm, prompt=prompt, stream=True)
    for chunk in res:
        print(chunk.get('response'), end='', flush=True)


    print("\ntest for current embedding function")
    embedder = "nomic-embed-text"
    embedding = ollama_client.embed(model=embedder,input="hi")
    print(embedding.get('embeddings'))
    print("\ntest for legacy embedding function")
    embedder = "nomic-embed-text"
    embedding = ollama_client.embeddings(model=embedder,prompt="hi")
    print(embedding.get('embedding'))


    print("\ntest fetching for model list")
    for model in ollama_client.list().get('models'):
        print(model.get('model'))
    print("\ntest fetching for running model list")
    for model in ollama_client.ps().get('models'):
        print(model.get('model'))
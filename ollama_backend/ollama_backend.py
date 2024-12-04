import modal
import subprocess
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import time
# Setup the model store volume to avoid repeated downloads.
volume = modal.Volume.from_name("ollama-store", create_if_missing=True)
model_store_path = "/vol/models"
mount = modal.Mount.from_local_dir(local_path='modelfiles', remote_path="modelfiles")

image = (modal.Image
        .debian_slim()
        .apt_install("curl")
        .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
        .run_commands("apt remove -y curl")
        .pip_install("ollama")
        .env({'OLLAMA_MODELS': model_store_path})
        .pip_install("fastapi[standard]")
        .pip_install("pydantic")
        .run_function(init_and_setup))


app = modal.App(
    name="modal-ollama"
)

def serve_ollama():
    """Ensure Ollama server is running."""
    subprocess.Popen(["ollama", "serve"])

class Payload(BaseModel):
    model: str
    messages: list = []
    prompt: str = ""

@app.cls(
    gpu='T4',
    allow_concurrent_inputs=10,
    volumes={model_store_path: volume},
    image=images["ollama"],
    container_idle_timeout=60,
)
class Ollama:
    model: str = modal.parameter(init=True)

    @modal.enter()
    def init_model(self):
        """Start the Ollama server."""
        print("Starting server")
        serve_ollama()

    @modal.method()
    def warmup(self):
        """Warmup the model."""
        ollama.generate(self.model)
        return {"status": "ok"}

    @modal.method()
    def list(self):
        models =[model['model'].split(':')[0] for model in ollama.list().models]
        return models if models else print("No models found")

    @modal.method()
    def chat(self, messages):
        """Handle chat interaction and stream the response."""
        stream = ollama.chat(self.model, messages, stream=True)
        for chunk in stream:
            yield chunk['message']['content']

    @modal.method()
    def generate(self, prompt):
        """Generate a response from the given user prompt."""
        for chunk in ollama.generate(model=self.model, prompt=prompt, stream=True):
            yield chunk['response']
            print(f"Generated: {chunk['response']}")

@app.function(volumes={model_store_path: volume}, timeout=60 * 30, image=image)
def model_download(repo_id: str):
    """Download the model from the Ollama server."""
    serve_ollama()
    time.sleep(1)
    print('pulling from ollama')
    ollama.create(repo_id)
    volume.commit()

@app.function(volumes={model_store_path: volume}, mounts=[mount], timeout=60 * 30, image=image)
def model_create(model_name: str):
    """Download the model from the Ollama server."""
    import os
    serve_ollama()
    time.sleep(1)
    print('creating new model')
    path = '/modelfiles/'+model_name
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file for {model_name} not found in {path}")
    #use local paths from mounted local directory
    ollama.create(model=model_name, path=path)
    volume.commit()

@app.function(volumes={model_store_path: volume}, allow_concurrent_inputs=1000, image=images["ollama"])
@modal.web_endpoint(method="POST", label="chat")
async def chat(payload: Payload):
    """Handle chat interaction and stream the response."""
    model = Ollama(model=payload.model)
    return StreamingResponse(model.chat.remote_gen(payload.messages), media_type="text/event-stream")

@app.function(volumes={model_store_path: volume}, allow_concurrent_inputs=1000, image=images["ollama"])
@modal.web_endpoint(method="POST", label="generate")
async def generate(payload: Payload):
    """Generate a response from the given user prompt."""
    model = Ollama(model=payload.model)
    return StreamingResponse(model.generate.remote_gen(payload.prompt), media_type="text/event-stream")

@app.function(volumes={model_store_path: volume}, allow_concurrent_inputs=1000, image=images["ollama"])
@modal.web_endpoint(method="POST", label="warmup")
def warmup(payload: Payload):
    """Warmup the model."""
    model = Ollama(model=payload.model)
    return model.warmup()


@app.local_entrypoint()
def init_and_setup():
    '''Initialize the server and download the models on deployment'''
    baseModel = "llama3.2"
    client = Ollama(model=baseModel)
    downloaded_models = client.list.remote()

    defaults = {
        "baseModel": baseModel,
        "embedModel": "nomic-embed-text", 
    }
    for key in defaults.keys():
        model_name = defaults[key]
        if model_name in downloaded_models:
            print(f"{model_name} already downloaded")
        else:
            print(f"{model_name} does not exist, downloading")
            model_download.remote(model_name)
    print('base models downloaded')
    derivatives = {
        "chatbotModel":"C3",
        "intentClassifier":"intentClassifier"
    }
    for key in derivatives.keys():
        model_name = derivatives[key]
        print(f"updating {model_name} with latest configs from file")
        model_create.remote(model_name)
    print('derived models created')
    print('Server initialized and running')

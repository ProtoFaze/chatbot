import modal
import subprocess
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import time
# Setup the model store volume to avoid repeated downloads.
volume = modal.Volume.from_name("ollama-store", create_if_missing=True)
model_store_path = "/vol/models"

images = {
    "ollama": modal.Image.debian_slim()
        .apt_install("curl")
        .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
        .run_commands("apt remove -y curl")
        .pip_install("ollama")
        .env({'OLLAMA_MODELS': model_store_path})
    # "api": modal.Image.debian_slim()
        .pip_install("fastapi[standard]")
        .pip_install("pydantic")
}

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
    def create(self, model, path):
        return ollama.create(model=model, path=path)

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

    
    
@app.function(volumes={model_store_path: volume}, timeout=60 * 30, image=images["ollama"])
def model_download(repo_id: str):
    """Download the model from the Ollama server."""
    serve_ollama()
    time.sleep(1)
    print('pulling from ollama')
    ollama.pull(repo_id)
    volume.commit()

@app.function(volumes={model_store_path: volume}, allow_concurrent_inputs=1000, image=images["ollama"])
@modal.web_endpoint(method="POST", label="chat")
def chat(payload: Payload):
    """Handle chat interaction and stream the response."""
    model = Ollama(model=payload.model)
    return StreamingResponse(model.chat.remote_gen(payload.messages), media_type="application/json")

@app.function(volumes={model_store_path: volume}, allow_concurrent_inputs=1000, image=images["ollama"])
@modal.web_endpoint(method="POST", label="generate")
def generate(payload: Payload):
    """Generate a response from the given user prompt."""
    model = Ollama(model=payload.model)
    return StreamingResponse(model.generate.remote_gen(payload.prompt), media_type="text/event-stream")

@app.function(volumes={model_store_path: volume}, allow_concurrent_inputs=1000, image=images["ollama"])
@modal.web_endpoint(method="POST", label="warmup")
def warmup(payload: Payload):
    """Warmup the model."""
    model = Ollama(model=payload.model)
    return model.warmup()

if __name__ == "__main__":
    with app.run():    
        baseModel = "llama3.2"
        client = Ollama(model=baseModel)
        defaults = {
            "baseModel": baseModel,
            "embedModel": "nomic-embed-text", 
        }
        downloaded_models = client.list.remote()
        for key in defaults.keys():
            model_name = defaults[key]
            if model_name in downloaded_models:
                print(f"{model_name} already downloaded")
            else:
                print(f"{model_name} does not exist, downloading")
                model_download.remote(model_name)
        print('Server initialized and running')

        
        
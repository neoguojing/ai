import threading
import time
import weakref

class ModelTimeoutManager:
    def __init__(self, timeout=60):
        self.timeout = timeout
        self.models = {}
        self.lock = threading.Lock()
        self.check_thread = threading.Thread(target=self._check_timeout)
        self.check_thread.daemon = True
        self.check_thread.start()

    def set_model(self, model_name, model):
        with self.lock:
            self.models[model_name] = {
                "model": weakref.ref(model),
                "last_access_time": time.time()
            }

    def get_model(self, model_name):
        with self.lock:
            if model_name in self.models:
                self.models[model_name]["last_access_time"] = time.time()
                return self.models[model_name]["model"]()
            return None

    def _check_timeout(self):
        while True:
            time.sleep(10)
            with self.lock:
                current_time = time.time()
                for model_name in list(self.models.keys()):
                    if current_time - self.models[model_name]["last_access_time"] > self.timeout:
                        model = self.models[model_name]["model"]()
                        if model is not None:
                            del model
                        del self.models[model_name]
                        print(f"Model '{model_name}' released due to timeout")
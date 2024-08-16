import sys
sys.path.append("..")
from ui import create_ui


if __name__ == "__main__":
    demo = create_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0",root_path="/algos")
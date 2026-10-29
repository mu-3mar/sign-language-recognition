import uvicorn

from slr.api import app
from slr.config import DEFAULT_HOST, DEFAULT_PORT


if __name__ == "__main__":
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)

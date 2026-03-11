#locust -f locustfile.py
import base64
import numpy as np
import cv2
from locust import HttpUser, task, between, TaskSet, User
from locust import events
from locust.runners import MasterRunner, WorkerRunner

# Create a shared dummy image
def create_dummy_image():
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.png', dummy_image)
    return base64.b64encode(buffer).decode('utf-8')

# Define tasks for each user
class SignLanguageTasks(TaskSet):
    def on_start(self):
        self.b64_image = create_dummy_image()

    @task(4)  # The /ws endpoint has a higher frequency (weight = 4)
    def process_image(self):
        self.client.post("/ws", json={"image": self.b64_image}, name="POST /ws")

    @task(1)  # The /reset endpoint is less frequent (weight = 1)
    def reset_processor(self):
        self.client.post("/reset", name="POST /reset")

# Define user types for different load levels
class LightLoadUser(HttpUser):
    tasks = [SignLanguageTasks]
    wait_time = between(1, 3)  # Wait between 1 to 3 seconds
    host = "http://20.121.41.74:5000"

class MediumLoadUser(HttpUser):
    tasks = [SignLanguageTasks]
    wait_time = between(0.5, 2)  # Shorter wait to simulate medium load
    host = "http://20.121.41.74:5000"

class HeavyLoadUser(HttpUser):
    tasks = [SignLanguageTasks]
    wait_time = between(0.2, 1)  # Very short wait to simulate heavy load
    host = "http://20.121.41.74:5000"

# Print summary statistics when the test ends
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("Performance test finished!")
    print(f"Total requests: {environment.stats.total.num_requests}")
    print(f"Total failures: {environment.stats.total.num_failures}")
    print(f"Average response time (ms): {environment.stats.total.avg_response_time:.2f}")
    print(f"Max response time (ms): {environment.stats.total.max_response_time:.2f}")

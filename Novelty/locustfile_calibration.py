"""Locust user whose inter-request wait is set per calibration stage."""
import os
from locust import HttpUser, task, between


class CalibrationUser(HttpUser):
    wait_time = between(
        float(os.getenv("LOCUST_WAIT_MIN", "0.1")),
        float(os.getenv("LOCUST_WAIT_MAX", "0.2")),
    )

    @task
    def predict(self):
        self.client.get("/predict?area=2500&rooms=4&age=8")

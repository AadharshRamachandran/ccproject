from locust import HttpUser, task, between
class HousePriceUser(HttpUser):
    wait_time=between(.01,.05)
    @task
    def predict(self): self.client.get('/predict?area=2500&rooms=4&age=8')

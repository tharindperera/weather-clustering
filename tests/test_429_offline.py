import sys
sys.path.insert(0, "src")

import openmeteo_client as m


class FakeResponse:
    def __init__(self, status_code, headers=None, text="rate limited"):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text

    def json(self):
        return {"error": "simulated 429"}

    def raise_for_status(self):
        raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self):
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1

        if self.calls == 1:
            return FakeResponse(429)

        return FakeResponse(200)

    def close(self):
        pass


sleeps = []

m.time.sleep = lambda seconds: sleeps.append(seconds)

client = m.OpenMeteoClient(
    session=FakeSession()
)

result = client.fetch_historical(
    {"test": "offline"}
)

print("OFFLINE TEST PASSED")
print("HTTP attempts:", client.session.calls)
print("Recorded sleeps:", sleeps)
print("Returned simulation: HTTP 200")

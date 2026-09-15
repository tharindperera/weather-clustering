import sys
sys.path.insert(0, "src")

import openmeteo_client as m


class FakeResponse:
    status_code = 429
    headers = {}
    text = "simulated persistent rate limit"

    def json(self):
        return {"error": "simulated persistent 429"}


class FakeSession:
    def __init__(self):
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return FakeResponse()

    def close(self):
        pass


sleeps = []

m.time.sleep = lambda seconds: sleeps.append(seconds)

client = m.OpenMeteoClient(
    session=FakeSession()
)

try:
    client.fetch_historical({"test": "persistent-429"})
except RuntimeError as exc:
    print("EXPECTED FAILURE CAPTURED")
    print("Error:", exc)

print("HTTP attempts:", client.session.calls)
print("Recorded sleeps:", sleeps)

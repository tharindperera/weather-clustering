import sys
sys.path.insert(0, "src")

import openmeteo_client as m


class FakeResponse:
    status_code = 429
    headers = {}
    text = "hourly limit"

    def json(self):
        return {
            "error": True,
            "reason": (
                "Hourly API request limit exceeded. "
                "Please try again in the next hour."
            ),
        }


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
    client.fetch_historical(
        {"test": "hourly-limit"}
    )
except RuntimeError as exc:
    print("EXPECTED HOURLY-LIMIT STOP")
    print("Error:", exc)

print("HTTP attempts:", client.session.calls)
print("Recorded sleeps:", sleeps)

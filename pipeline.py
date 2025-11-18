from graph import graph, State

reviews = [
    {"text": "The app crashes when I open it", "date": "2025-10-05"},
    {"text": "I love the new dark mode feature", "date": "2025-10-10"},
    {"text": "Notifications are not working properly", "date": "2025-10-15"},
    {"text": "The recent update improved performance", "date": "2025-11-02"},
    {"text": "I found a bug in the login page", "date": "2025-11-07"},
    {"text": "App is very slow on my device", "date": "2025-11-12"},
    {"text": "The app design is sleek and intuitive", "date": "2025-12-03"},
    {"text": "I want a feature to export my data", "date": "2025-12-08"},
    {"text": "There are too many ads in the app", "date": "2025-12-14"},
    {"text": "Customer support responded very quickly", "date": "2026-01-05"}
]

state = {
    "reviews": reviews,
    "start_date": "2025-10-01",
    "end_date": "2026-01-05",
    "output": "output/report.csv"
}

def run_daily(reviews, state_dates):
    state["reviews"] = reviews
    state["start_date"], state["end_date"] = state_dates
    graph.invoke(state)

run_daily(reviews, ("2025-10-01", "2026-01-05"))

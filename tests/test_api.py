import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
from main import app, get_orchestrator
from agents import TravelOrchestratorAgent


@pytest.fixture
def mock_orchestrator():
    mock = MagicMock(spec=TravelOrchestratorAgent)
    mock.chat = AsyncMock()
    mock.weather_agent = MagicMock()
    mock.weather_agent.generate = AsyncMock()
    mock.itinerary_agent = MagicMock()
    mock.itinerary_agent.generate = AsyncMock()
    return mock


@pytest.fixture
def client(mock_orchestrator):
    app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_chat_endpoint(client, mock_orchestrator):
    mock_orchestrator.chat.return_value = {
        "reply": "Warm intro to Paris",
        "plan": {
            "destination": "Paris",
            "days": 5,
            "budget": "Mid-range",
            "interests": ["sightseeing"],
            "travelers": 1,
            "weather": {
                "temperature": "72°F",
                "conditions": "Sunny",
                "best_time_to_visit": "Spring",
                "travel_tips": "Wear sunscreen",
            },
            "itinerary": {"destination": "Paris", "days": []},
            "hotels_food": {"hotels": [], "restaurants": []},
            "budget_details": {
                "currency": "USD",
                "budget_tier": {
                    "flights": "$500",
                    "hotels": "$300",
                    "food": "$200",
                    "activities": "$100",
                    "local_transport": "$50",
                    "total": "$1150",
                },
                "mid_range_tier": {
                    "flights": "$800",
                    "hotels": "$700",
                    "food": "$400",
                    "activities": "$250",
                    "local_transport": "$100",
                    "total": "$2250",
                },
                "luxury_tier": {
                    "flights": "$1500",
                    "hotels": "$2000",
                    "food": "$800",
                    "activities": "$600",
                    "local_transport": "$300",
                    "total": "$5200",
                },
            },
            "packing": {
                "clothes": [],
                "documents": [],
                "electronics": [],
                "medicines": [],
                "accessories": [],
            },
        },
        "session_id": "test_session",
    }

    response = client.post(
        "/chat",
        json={
            "message": "Plan a trip to Paris",
            "session_id": "test_session",
            "destination": "Paris",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "Warm intro to Paris"
    assert data["plan"]["destination"] == "Paris"
    mock_orchestrator.chat.assert_called_once_with(
        message="Plan a trip to Paris", session_id="test_session", destination="Paris"
    )


def test_weather_endpoint(client, mock_orchestrator):
    mock_weather_res = MagicMock()
    mock_weather_res.model_dump.return_value = {
        "temperature": "72°F",
        "conditions": "Sunny",
        "best_time_to_visit": "Spring",
        "travel_tips": "Wear sunscreen",
    }
    mock_orchestrator.weather_agent.generate.return_value = mock_weather_res

    response = client.post("/weather", json={"destination": "Paris"})
    assert response.status_code == 200
    data = response.json()
    assert data["temperature"] == "72°F"
    assert data["conditions"] == "Sunny"
    mock_orchestrator.weather_agent.generate.assert_called_once_with("Paris")


def test_itinerary_endpoint(client, mock_orchestrator):
    mock_itinerary_res = MagicMock()
    mock_itinerary_res.model_dump.return_value = {"destination": "Paris", "days": []}
    mock_orchestrator.itinerary_agent.generate.return_value = mock_itinerary_res

    response = client.post(
        "/itinerary",
        json={
            "destination": "Paris",
            "days": 5,
            "budget": "Mid-range",
            "interests": ["sightseeing"],
            "travelers": 1,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["destination"] == "Paris"
    mock_orchestrator.itinerary_agent.generate.assert_called_once_with(
        destination="Paris",
        days=5,
        budget="Mid-range",
        interests=["sightseeing"],
        travelers=1,
    )


def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "WanderAI" in response.text


def test_chat_endpoint_error(client, mock_orchestrator):
    mock_orchestrator.chat.side_effect = Exception("Chat error")
    response = client.post(
        "/chat", json={"message": "Hello", "session_id": "test_session"}
    )
    assert response.status_code == 500
    assert "An error occurred" in response.json()["detail"]


def test_weather_endpoint_error(client, mock_orchestrator):
    mock_orchestrator.weather_agent.generate.side_effect = Exception("Weather error")
    response = client.post("/weather", json={"destination": "Paris"})
    assert response.status_code == 500
    assert "Failed to retrieve weather" in response.json()["detail"]


def test_itinerary_endpoint_error(client, mock_orchestrator):
    mock_orchestrator.itinerary_agent.generate.side_effect = Exception(
        "Itinerary error"
    )
    response = client.post(
        "/itinerary",
        json={
            "destination": "Paris",
            "days": 5,
            "budget": "Mid-range",
            "interests": [],
            "travelers": 1,
        },
    )
    assert response.status_code == 500
    assert "Failed to generate itinerary" in response.json()["detail"]


def test_get_orchestrator_unavailable(client):
    app.dependency_overrides.clear()
    orig_orchestrator = app.state.orchestrator
    app.state.orchestrator = None

    try:
        response = client.post("/chat", json={"message": "hello"})
        assert response.status_code == 503
        assert "service is currently unavailable" in response.json()["detail"]
    finally:
        app.state.orchestrator = orig_orchestrator


def test_payload_too_large(client):
    large_payload = "a" * (1024 * 1024 + 100)  # > 1MB
    response = client.post(
        "/chat",
        content=large_payload,
        headers={"Content-Length": str(len(large_payload))},
    )
    assert response.status_code == 413
    assert "Payload Too Large" in response.text


def test_invalid_content_length(client):
    response = client.post(
        "/chat", content="foo", headers={"Content-Length": "not_an_int"}
    )
    assert response.status_code == 400
    assert "Invalid Content-Length Header" in response.text


@pytest.mark.asyncio
async def test_lifespan_exception():
    from main import lifespan

    mock_app = MagicMock()
    with patch("google.genai.Client", side_effect=Exception("Client init error")):
        async with lifespan(mock_app):
            pass
    assert mock_app.state.orchestrator is None


def test_get_orchestrator_success():
    from main import get_orchestrator

    mock_request = MagicMock()
    mock_orchestrator_instance = MagicMock()
    mock_request.app.state.orchestrator = mock_orchestrator_instance
    assert get_orchestrator(mock_request) is mock_orchestrator_instance

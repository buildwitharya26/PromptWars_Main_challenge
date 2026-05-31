import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock
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
                "travel_tips": "Wear sunscreen"
            },
            "itinerary": {
                "destination": "Paris",
                "days": []
            },
            "hotels_food": {
                "hotels": [],
                "restaurants": []
            },
            "budget_details": {
                "currency": "USD",
                "budget_tier": {"flights": "$500", "hotels": "$300", "food": "$200", "activities": "$100", "local_transport": "$50", "total": "$1150"},
                "mid_range_tier": {"flights": "$800", "hotels": "$700", "food": "$400", "activities": "$250", "local_transport": "$100", "total": "$2250"},
                "luxury_tier": {"flights": "$1500", "hotels": "$2000", "food": "$800", "activities": "$600", "local_transport": "$300", "total": "$5200"}
            },
            "packing": {
                "clothes": [],
                "documents": [],
                "electronics": [],
                "medicines": [],
                "accessories": []
            }
        },
        "session_id": "test_session"
    }

    response = client.post("/chat", json={
        "message": "Plan a trip to Paris",
        "session_id": "test_session",
        "destination": "Paris"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "Warm intro to Paris"
    assert data["plan"]["destination"] == "Paris"
    mock_orchestrator.chat.assert_called_once_with(
        message="Plan a trip to Paris",
        session_id="test_session",
        destination="Paris"
    )

def test_weather_endpoint(client, mock_orchestrator):
    mock_weather_res = MagicMock()
    mock_weather_res.model_dump.return_value = {
        "temperature": "72°F",
        "conditions": "Sunny",
        "best_time_to_visit": "Spring",
        "travel_tips": "Wear sunscreen"
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
    mock_itinerary_res.model_dump.return_value = {
        "destination": "Paris",
        "days": []
    }
    mock_orchestrator.itinerary_agent.generate.return_value = mock_itinerary_res
    
    response = client.post("/itinerary", json={
        "destination": "Paris",
        "days": 5,
        "budget": "Mid-range",
        "interests": ["sightseeing"],
        "travelers": 1
    })
    assert response.status_code == 200
    data = response.json()
    assert data["destination"] == "Paris"
    mock_orchestrator.itinerary_agent.generate.assert_called_once_with(
        destination="Paris",
        days=5,
        budget="Mid-range",
        interests=["sightseeing"],
        travelers=1
    )

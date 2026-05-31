import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from google import genai
from agents import (
    WeatherAgent, WeatherResponse,
    ItineraryAgent, ItineraryResponse,
    HotelAndFoodAgent, HotelFoodResponse,
    BudgetAgent, BudgetResponse,
    PackingAgent, PackingResponse
)

@pytest.fixture
def mock_genai_client():
    mock_client = MagicMock(spec=genai.Client)
    mock_client.aio = MagicMock()
    mock_client.aio.models = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock()
    return mock_client

@pytest.mark.asyncio
async def test_weather_agent(mock_genai_client):
    mock_res = MagicMock()
    mock_res.text = """
    {
        "temperature": "72°F",
        "conditions": "Sunny",
        "best_time_to_visit": "Spring",
        "travel_tips": "Wear sunscreen"
    }
    """
    mock_genai_client.aio.models.generate_content.return_value = mock_res
    
    with patch("agents.geocode_destination", new_callable=AsyncMock) as mock_geo:
        mock_geo.return_value = {
            "latitude": 48.8566,
            "longitude": 2.3522,
            "country_code": "FR",
            "is_us": False,
            "name": "Paris"
        }
        
        agent = WeatherAgent(mock_genai_client)
        res = await agent.generate("Paris")
        
        assert isinstance(res, WeatherResponse)
        assert res.temperature == "72°F"
        assert res.conditions == "Sunny"

@pytest.mark.asyncio
async def test_itinerary_agent(mock_genai_client):
    mock_res = MagicMock()
    mock_res.text = """
    {
        "destination": "Paris",
        "days": [
            {
                "day": 1,
                "morning": {
                    "time": "09:00 AM - 11:30 AM",
                    "activity": "Eiffel Tower Tour",
                    "location": "Eiffel Tower",
                    "description": "Visit the iconic Eiffel Tower."
                },
                "afternoon": {
                    "time": "12:00 PM - 03:00 PM",
                    "activity": "Louvre Museum Tour",
                    "location": "Louvre Museum",
                    "description": "See the Mona Lisa."
                },
                "evening": {
                    "time": "06:00 PM - 09:00 PM",
                    "activity": "Seine River Cruise",
                    "location": "Seine River",
                    "description": "Enjoy a scenic boat cruise."
                },
                "local_tips": ["Buy tickets online in advance"],
                "transportation": "Metro Line 6"
            }
        ]
    }
    """
    mock_genai_client.aio.models.generate_content.return_value = mock_res
    
    agent = ItineraryAgent(mock_genai_client)
    res = await agent.generate("Paris", 1, "Mid-range", ["sightseeing"], 1)
    
    assert isinstance(res, ItineraryResponse)
    assert res.destination == "Paris"
    assert len(res.days) == 1
    assert res.days[0].morning.activity == "Eiffel Tower Tour"

@pytest.mark.asyncio
async def test_hotel_and_food_agent(mock_genai_client):
    mock_res = MagicMock()
    mock_res.text = """
    {
        "hotels": [
            {
                "name": "Le Bristol",
                "description": "Luxury hotel near Champs-Elysees",
                "price_range": "$1000 - $1500",
                "tier": "Luxury",
                "rating": 4.9
            }
        ],
        "restaurants": [
            {
                "name": "Bistrot Paul Bert",
                "description": "Classic French bistro",
                "must_try_dishes": ["Steak frites", "Paris-Brest"],
                "specialty": "French Bistro"
            }
        ]
    }
    """
    mock_genai_client.aio.models.generate_content.return_value = mock_res
    
    agent = HotelAndFoodAgent(mock_genai_client)
    res = await agent.generate("Paris")
    
    assert isinstance(res, HotelFoodResponse)
    assert len(res.hotels) == 1
    assert res.hotels[0].name == "Le Bristol"
    assert len(res.restaurants) == 1
    assert res.restaurants[0].name == "Bistrot Paul Bert"

@pytest.mark.asyncio
async def test_budget_agent(mock_genai_client):
    mock_res = MagicMock()
    mock_res.text = """
    {
        "currency": "USD",
        "budget_tier": {
            "flights": "$500",
            "hotels": "$300",
            "food": "$200",
            "activities": "$100",
            "local_transport": "$50",
            "total": "$1150"
        },
        "mid_range_tier": {
            "flights": "$800",
            "hotels": "$700",
            "food": "$400",
            "activities": "$250",
            "local_transport": "$100",
            "total": "$2250"
        },
        "luxury_tier": {
            "flights": "$1500",
            "hotels": "$2000",
            "food": "$800",
            "activities": "$600",
            "local_transport": "$300",
            "total": "$5200"
        }
    }
    """
    mock_genai_client.aio.models.generate_content.return_value = mock_res
    
    agent = BudgetAgent(mock_genai_client)
    res = await agent.generate("Paris", 5)
    
    assert isinstance(res, BudgetResponse)
    assert res.mid_range_tier.total == "$2250"

@pytest.mark.asyncio
async def test_packing_agent(mock_genai_client):
    mock_res = MagicMock()
    mock_res.text = """
    {
        "clothes": [{"item": "Jacket", "quantity": "1", "necessity": "Required"}],
        "documents": [{"item": "Passport", "quantity": "1", "necessity": "Required"}],
        "electronics": [{"item": "Phone Charger", "quantity": "1", "necessity": "Required"}],
        "medicines": [{"item": "First Aid Kit", "quantity": "1", "necessity": "Optional"}],
        "accessories": [{"item": "Sunglasses", "quantity": "1", "necessity": "Optional"}]
    }
    """
    mock_genai_client.aio.models.generate_content.return_value = mock_res
    
    agent = PackingAgent(mock_genai_client)
    res = await agent.generate("Paris", 5, "Sunny")
    
    assert isinstance(res, PackingResponse)
    assert res.clothes[0].item == "Jacket"

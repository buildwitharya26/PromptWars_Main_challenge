import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from google import genai
from agents import (
    WeatherAgent,
    WeatherResponse,
    ItineraryAgent,
    ItineraryResponse,
    HotelAndFoodAgent,
    HotelFoodResponse,
    BudgetAgent,
    BudgetResponse,
    PackingAgent,
    PackingResponse,
    TravelOrchestratorAgent,
    TravelPlan,
    CostBreakdown,
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
            "name": "Paris",
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


def test_session_service():
    from agents import InMemorySessionService

    service = InMemorySessionService()
    assert service.get_history("session-1") == []
    service.add_message("session-1", "user", "hello")
    assert service.get_history("session-1") == [{"role": "user", "content": "hello"}]
    assert service.get_plan("session-1") is None

    plan = TravelPlan(
        destination="Paris",
        days=5,
        budget="Mid-range",
        interests=[],
        travelers=1,
        weather=WeatherResponse(
            temperature="72°F",
            conditions="Sunny",
            best_time_to_visit="Spring",
            travel_tips="Wear sunscreen",
        ),
        itinerary=ItineraryResponse(destination="Paris", days=[]),
        hotels_food=HotelFoodResponse(hotels=[], restaurants=[]),
        budget_details=BudgetResponse(
            currency="USD",
            budget_tier=CostBreakdown(
                flights="",
                hotels="",
                food="",
                activities="",
                local_transport="",
                total="",
            ),
            mid_range_tier=CostBreakdown(
                flights="",
                hotels="",
                food="",
                activities="",
                local_transport="",
                total="",
            ),
            luxury_tier=CostBreakdown(
                flights="",
                hotels="",
                food="",
                activities="",
                local_transport="",
                total="",
            ),
        ),
        packing=PackingResponse(
            clothes=[], documents=[], electronics=[], medicines=[], accessories=[]
        ),
    )
    service.set_plan("session-1", plan)
    assert service.get_plan("session-1") == plan
    service.clear("session-1")
    assert service.get_history("session-1") == []
    assert service.get_plan("session-1") is None


@pytest.mark.asyncio
async def test_travel_orchestrator_agent_create_plan(mock_genai_client):
    orchestrator = TravelOrchestratorAgent(mock_genai_client)

    mock_intent_res = MagicMock()
    mock_intent_res.text = """
    {
        "action": "CREATE_PLAN",
        "destination": "Paris",
        "days": 5,
        "budget": "Mid-range",
        "interests": ["sightseeing"],
        "travelers": 1
    }
    """

    orchestrator.weather_agent.generate = AsyncMock(
        return_value=WeatherResponse(
            temperature="72°F",
            conditions="Sunny",
            best_time_to_visit="Spring",
            travel_tips="Wear sunscreen",
        )
    )
    orchestrator.itinerary_agent.generate = AsyncMock(
        return_value=ItineraryResponse(destination="Paris", days=[])
    )
    orchestrator.hotel_food_agent.generate = AsyncMock(
        return_value=HotelFoodResponse(hotels=[], restaurants=[])
    )
    orchestrator.budget_agent.generate = AsyncMock(
        return_value=BudgetResponse(
            currency="USD",
            budget_tier=CostBreakdown(
                flights="",
                hotels="",
                food="",
                activities="",
                local_transport="",
                total="",
            ),
            mid_range_tier=CostBreakdown(
                flights="",
                hotels="",
                food="",
                activities="",
                local_transport="",
                total="",
            ),
            luxury_tier=CostBreakdown(
                flights="",
                hotels="",
                food="",
                activities="",
                local_transport="",
                total="",
            ),
        )
    )
    orchestrator.packing_agent.generate = AsyncMock(
        return_value=PackingResponse(
            clothes=[], documents=[], electronics=[], medicines=[], accessories=[]
        )
    )

    mock_summary_res = MagicMock()
    mock_summary_res.text = "Here is your awesome travel plan to Paris!"

    mock_genai_client.aio.models.generate_content.side_effect = [
        mock_intent_res,
        mock_summary_res,
    ]

    res = await orchestrator.chat("Plan a trip to Paris", "session-123")

    assert res["reply"] == "Here is your awesome travel plan to Paris!"
    assert res["plan"]["destination"] == "Paris"
    assert res["session_id"] == "session-123"


@pytest.mark.asyncio
async def test_travel_orchestrator_agent_chat_conversational(mock_genai_client):
    orchestrator = TravelOrchestratorAgent(mock_genai_client)

    mock_intent_res = MagicMock()
    mock_intent_res.text = """
    {
        "action": "CHAT",
        "conversational_reply": "Hello! How can I help you today?"
    }
    """
    mock_genai_client.aio.models.generate_content.return_value = mock_intent_res

    res = await orchestrator.chat("Hello", "session-123")

    assert res["reply"] == "Hello! How can I help you today?"
    assert res["plan"] is None
    assert res["session_id"] == "session-123"


@pytest.mark.asyncio
async def test_travel_orchestrator_agent_modify_plan(mock_genai_client):
    orchestrator = TravelOrchestratorAgent(mock_genai_client)

    existing_plan = TravelPlan(
        destination="Paris",
        days=5,
        budget="Mid-range",
        interests=["sightseeing"],
        travelers=1,
        weather=WeatherResponse(
            temperature="72°F",
            conditions="Sunny",
            best_time_to_visit="Spring",
            travel_tips="Wear sunscreen",
        ),
        itinerary=ItineraryResponse(destination="Paris", days=[]),
        hotels_food=HotelFoodResponse(hotels=[], restaurants=[]),
        budget_details=BudgetResponse(
            currency="USD",
            budget_tier=CostBreakdown(
                flights="",
                hotels="",
                food="",
                activities="",
                local_transport="",
                total="",
            ),
            mid_range_tier=CostBreakdown(
                flights="",
                hotels="",
                food="",
                activities="",
                local_transport="",
                total="",
            ),
            luxury_tier=CostBreakdown(
                flights="",
                hotels="",
                food="",
                activities="",
                local_transport="",
                total="",
            ),
        ),
        packing=PackingResponse(
            clothes=[], documents=[], electronics=[], medicines=[], accessories=[]
        ),
    )
    orchestrator.session_service.set_plan("session-123", existing_plan)

    mock_intent_res = MagicMock()
    mock_intent_res.text = """
    {
        "action": "MODIFY_PLAN",
        "modification_instructions": "add museums to day 2"
    }
    """

    mock_revision_res = MagicMock()
    mock_revision_res.text = """
    {
        "destination": "Paris",
        "days": []
    }
    """

    mock_genai_client.aio.models.generate_content.side_effect = [
        mock_intent_res,
        mock_revision_res,
    ]
    orchestrator.packing_agent.generate = AsyncMock(
        return_value=PackingResponse(
            clothes=[], documents=[], electronics=[], medicines=[], accessories=[]
        )
    )

    res = await orchestrator.chat("please add museums", "session-123")

    assert "updated" in res["reply"]
    assert res["plan"]["destination"] == "Paris"
    assert res["session_id"] == "session-123"


@pytest.mark.asyncio
async def test_travel_orchestrator_agent_exception(mock_genai_client):
    orchestrator = TravelOrchestratorAgent(mock_genai_client)

    mock_intent_res = MagicMock()
    mock_intent_res.text = """
    {
        "action": "CREATE_PLAN",
        "destination": "Paris"
    }
    """
    mock_genai_client.aio.models.generate_content.return_value = mock_intent_res
    orchestrator.weather_agent.generate = AsyncMock(
        side_effect=RuntimeError("API Error")
    )

    res = await orchestrator.chat("Plan a trip to Paris", "session-123")

    assert "Sorry, I encountered an issue planning your trip" in res["reply"]
    assert res["plan"] is None


@pytest.mark.asyncio
async def test_travel_orchestrator_agent_modify_error(mock_genai_client):
    orchestrator = TravelOrchestratorAgent(mock_genai_client)

    existing_plan = TravelPlan(
        destination="Paris",
        days=5,
        budget="Mid-range",
        interests=["sightseeing"],
        travelers=1,
        weather=WeatherResponse(
            temperature="72°F",
            conditions="Sunny",
            best_time_to_visit="Spring",
            travel_tips="Wear sunscreen",
        ),
        itinerary=ItineraryResponse(destination="Paris", days=[]),
        hotels_food=HotelFoodResponse(hotels=[], restaurants=[]),
        budget_details=BudgetResponse(
            currency="USD",
            budget_tier=CostBreakdown(
                flights="",
                hotels="",
                food="",
                activities="",
                local_transport="",
                total="",
            ),
            mid_range_tier=CostBreakdown(
                flights="",
                hotels="",
                food="",
                activities="",
                local_transport="",
                total="",
            ),
            luxury_tier=CostBreakdown(
                flights="",
                hotels="",
                food="",
                activities="",
                local_transport="",
                total="",
            ),
        ),
        packing=PackingResponse(
            clothes=[], documents=[], electronics=[], medicines=[], accessories=[]
        ),
    )
    orchestrator.session_service.set_plan("session-123", existing_plan)

    mock_intent_res = MagicMock()
    mock_intent_res.text = """
    {
        "action": "MODIFY_PLAN",
        "modification_instructions": "add museums to day 2"
    }
    """

    mock_revision_res = MagicMock()
    mock_revision_res.text = None

    mock_genai_client.aio.models.generate_content.side_effect = [
        mock_intent_res,
        mock_revision_res,
    ]

    res = await orchestrator.chat("please add museums", "session-123")
    assert "encountered an error" in res["reply"]


@pytest.mark.asyncio
async def test_weather_agent_us(mock_genai_client):
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

    with (
        patch("agents.geocode_destination", new_callable=AsyncMock) as mock_geo,
        patch("agents.get_us_weather", new_callable=AsyncMock) as mock_weather,
    ):
        mock_geo.return_value = {
            "latitude": 47.6062,
            "longitude": -122.3321,
            "country_code": "US",
            "is_us": True,
            "name": "Seattle",
        }
        mock_weather.return_value = "Rainy today."

        agent = WeatherAgent(mock_genai_client)
        await agent.generate("Seattle")

        mock_weather.assert_called_once_with(47.6062, -122.3321)


@pytest.mark.asyncio
async def test_travel_orchestrator_intent_none(mock_genai_client):
    orchestrator = TravelOrchestratorAgent(mock_genai_client)
    mock_res = MagicMock()
    mock_res.text = None
    mock_genai_client.aio.models.generate_content.return_value = mock_res

    with pytest.raises(
        ValueError, match="Model returned empty text for intent analysis"
    ):
        await orchestrator._determine_intent("hello", [], False)

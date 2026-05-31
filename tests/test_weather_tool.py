import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from tools import geocode_destination, get_us_weather

@pytest.mark.asyncio
async def test_geocode_destination_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "latitude": 48.8566,
                "longitude": 2.3522,
                "country_code": "FR",
                "name": "Paris"
            }
        ]
    }
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        
        result = await geocode_destination("Paris")
        assert result is not None
        assert result["latitude"] == 48.8566
        assert result["longitude"] == 2.3522
        assert result["is_us"] is False
        assert result["name"] == "Paris"

@pytest.mark.asyncio
async def test_get_us_weather_success():
    mock_points_res = MagicMock()
    mock_points_res.status_code = 200
    mock_points_res.json.return_value = {
        "properties": {
            "forecast": "https://api.weather.gov/gridpoints/FOO/12,34/forecast"
        }
    }
    
    mock_forecast_res = MagicMock()
    mock_forecast_res.status_code = 200
    mock_forecast_res.json.return_value = {
        "properties": {
            "periods": [
                {
                    "name": "Today",
                    "temperature": 72,
                    "temperatureUnit": "F",
                    "shortForecast": "Sunny",
                    "detailedForecast": "Sunny all day"
                }
            ]
        }
    }
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = [mock_points_res, mock_forecast_res]
        
        result = await get_us_weather(37.7749, -122.4194)
        assert "Today" in result
        assert "72°F" in result
        assert "Sunny" in result

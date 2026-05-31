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
                "name": "Paris",
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
                    "detailedForecast": "Sunny all day",
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


@pytest.mark.asyncio
async def test_geocode_destination_cache():
    from tools import _geocode_cache

    _geocode_cache.clear()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "latitude": 35.6762,
                "longitude": 139.6503,
                "country_code": "JP",
                "name": "Tokyo",
            }
        ]
    }

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response

        # First call (should hit mock API)
        res1 = await geocode_destination("Tokyo")
        assert res1["name"] == "Tokyo"
        assert mock_get.call_count == 1

        # Second call (should hit cache, no API call)
        res2 = await geocode_destination("Tokyo")
        assert res2["name"] == "Tokyo"
        assert mock_get.call_count == 1  # Still 1, meaning cached


@pytest.mark.asyncio
async def test_geocode_destination_no_results():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": []}

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        result = await geocode_destination("InvalidCityNameThatWillNotGeocode")
        assert result is None


@pytest.mark.asyncio
async def test_geocode_destination_non_200():
    mock_response = MagicMock()
    mock_response.status_code = 500

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        result = await geocode_destination("Paris")
        assert result is None


@pytest.mark.asyncio
async def test_geocode_destination_exception():
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = Exception("Connection error")
        result = await geocode_destination("Paris")
        assert result is None


@pytest.mark.asyncio
async def test_get_us_weather_points_failed():
    mock_response = MagicMock()
    mock_response.status_code = 500

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        result = await get_us_weather(37.7749, -122.4194)
        assert "US weather details currently unavailable" in result


@pytest.mark.asyncio
async def test_get_us_weather_forecast_url_missing():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"properties": {}}

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        result = await get_us_weather(37.7749, -122.4194)
        assert "forecast URL missing" in result


@pytest.mark.asyncio
async def test_get_us_weather_forecast_failed():
    mock_points_res = MagicMock()
    mock_points_res.status_code = 200
    mock_points_res.json.return_value = {
        "properties": {
            "forecast": "https://api.weather.gov/gridpoints/FOO/12,34/forecast"
        }
    }

    mock_forecast_res = MagicMock()
    mock_forecast_res.status_code = 500

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = [mock_points_res, mock_forecast_res]
        result = await get_us_weather(37.7749, -122.4194)
        assert "US weather forecast details currently unavailable" in result


@pytest.mark.asyncio
async def test_get_us_weather_exception():
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = Exception("API error")
        result = await get_us_weather(37.7749, -122.4194)
        assert "Error connecting to api.weather.gov" in result


@pytest.mark.asyncio
async def test_get_us_weather_no_periods():
    mock_points_res = MagicMock()
    mock_points_res.status_code = 200
    mock_points_res.json.return_value = {
        "properties": {
            "forecast": "https://api.weather.gov/gridpoints/FOO/12,34/forecast"
        }
    }

    mock_forecast_res = MagicMock()
    mock_forecast_res.status_code = 200
    mock_forecast_res.json.return_value = {"properties": {"periods": []}}

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = [mock_points_res, mock_forecast_res]
        result = await get_us_weather(37.7749, -122.4194)
        assert "No weather forecast periods returned" in result

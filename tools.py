import logging
import httpx
from typing import Dict, Any, Optional

logger = logging.getLogger("wanderai.tools")

_geocode_cache: Dict[str, Dict[str, Any]] = {}


async def geocode_destination(destination: str) -> Optional[Dict[str, Any]]:
    """Geocodes a destination name to get latitude, longitude, and country details.

    Uses Open-Meteo's public geocoding API which does not require an API key.

    Args:
        destination: Name of the city/destination.

    Returns:
        Dict with latitude, longitude, country_code, is_us, and name, or None if failed.
    """
    dest_key = destination.strip().lower()
    if dest_key in _geocode_cache:
        logger.info(f"Geocoding cache hit for: {destination}")
        return _geocode_cache[dest_key]

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params: Dict[str, Any] = {
        "name": destination,
        "count": 1,
        "language": "en",
        "format": "json",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            logger.info(f"Geocoding destination: {destination}")
            response = await client.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results")
                if results:
                    result = results[0]
                    country_code = result.get("country_code", "").upper()
                    info = {
                        "latitude": result.get("latitude"),
                        "longitude": result.get("longitude"),
                        "country_code": country_code,
                        "is_us": country_code == "US",
                        "name": result.get("name"),
                    }
                    logger.info(f"Geocoding success: {destination} -> {info}")
                    _geocode_cache[dest_key] = info
                    return info
                else:
                    logger.warning(f"No geocoding results found for {destination}")
            else:
                logger.error(
                    f"Geocoding API returned status {response.status_code} for {destination}"
                )
        except Exception as e:
            logger.exception(f"Exception during geocoding for {destination}: {e}")

    return None


async def get_us_weather(latitude: float, longitude: float) -> str:
    """Fetches real-time weather forecast from api.weather.gov for a US location.

    Args:
        latitude: Latitude coordinate.
        longitude: Longitude coordinate.

    Returns:
        A string summarizing the weather forecast, or error message.
    """
    headers = {
        "User-Agent": "WanderAI/1.0 (contact@wanderai.example.com)",
        "Accept": "application/ld+json",
    }

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        try:
            # 1. Fetch gridpoints metadata URL
            points_url = (
                f"https://api.weather.gov/points/{latitude:.4f},{longitude:.4f}"
            )
            logger.info(f"Fetching weather points from: {points_url}")
            response = await client.get(points_url, headers=headers)
            if response.status_code != 200:
                logger.error(
                    f"api.weather.gov points failed: {response.status_code} {response.text}"
                )
                return "US weather details currently unavailable from weather.gov (points fetch failed)."

            data = response.json()
            forecast_url = data.get("properties", {}).get("forecast")
            if not forecast_url:
                logger.error(
                    f"No forecast URL found in weather.gov points response: {data}"
                )
                return (
                    "US weather details currently unavailable (forecast URL missing)."
                )

            # 2. Fetch the actual weather forecast periods
            logger.info(f"Fetching weather forecast from: {forecast_url}")
            forecast_response = await client.get(forecast_url, headers=headers)
            if forecast_response.status_code != 200:
                logger.error(
                    f"api.weather.gov forecast failed: {forecast_response.status_code}"
                )
                return "US weather forecast details currently unavailable."

            forecast_data = forecast_response.json()
            periods = forecast_data.get("properties", {}).get("periods", [])
            if not periods:
                return "No weather forecast periods returned by weather.gov."

            weather_summary = []
            for period in periods[
                :3
            ]:  # Capture first 3 periods (e.g. Today, Tonight, Tomorrow)
                name = period.get("name", "Forecast")
                temp = period.get("temperature", "N/A")
                unit = period.get("temperatureUnit", "F")
                detailed = period.get("detailedForecast", "")
                short_forecast = period.get("shortForecast", "")

                weather_summary.append(
                    f"{name}: {temp}°{unit}. {short_forecast}. {detailed}"
                )

            return "\n".join(weather_summary)

        except Exception as e:
            logger.exception(f"Exception during weather.gov fetch: {e}")
            return f"Error connecting to api.weather.gov: {str(e)}"

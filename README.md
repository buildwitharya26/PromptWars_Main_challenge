# WanderAI - Your AI Travel Planner

WanderAI is a production-grade, premium travel planning web application. It uses a **multi-agent orchestrator architecture** with Vertex AI / Gemini 2.5 Flash models to generate customized, highly accurate travel plans, weather forecasts, itineraries, accommodation suggestions, and packing lists.

---

## 🌟 Key Features

- **Multi-Turn Chat Assistant**: Engage in conversational chat to build and fine-tune your custom itinerary.
- **Glassmorphic SPA Dashboard**: Visual presentations of trip details including:
  - **Dynamic Weather Cards** (US weather from `weather.gov`, international from Google Search grounding).
  - **Interactive Itinerary Timelines** (expandable morning, afternoon, and evening slots).
  - **Accommodations & Dining** (budget/luxury tier hotels and specialty restaurant listings).
  - **Expense Gauges** (comparing costs for flights, hotels, food, activities, and transport).
  - **Packing Checklists** (interactive lists grouped by category).
- **Google Search Grounding**: Leverages Gemini's built-in Search Tool to fetch real-time hotel ratings, weather conditions, and tourist landmarks.

---

## 🏗️ Architecture Overview

The backend uses a master-specialist agent pattern:
1. **TravelOrchestratorAgent**: Evaluates message intent (`CREATE_PLAN`, `MODIFY_PLAN`, `CHAT`) and orchestrates specialist agents.
2. **WeatherAgent**: Geocodes destinations and provides current weather.
3. **ItineraryAgent**: Generates detailed day-by-day sightseeing activities.
4. **HotelAndFoodAgent**: Recommends hotels across three tiers and dining spots.
5. **BudgetAgent**: Provides a breakdown of costs across different categories.
6. **PackingAgent**: Curates packing checklists based on duration and weather.

---

## 🚀 Setup & Execution

### Prerequisites
- Python 3.11+
- Google Cloud Project with Vertex AI API enabled.
- Application Default Credentials (ADC) or Service Account key setup.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Create a `.env` file in the root directory:
```env
GOOGLE_GENAI_USE_VERTEXAI=True
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
LOG_LEVEL=INFO
```

### 3. Run Locally
Start the FastAPI development server:
```bash
python -m uvicorn main:app --reload --port 8000
```
Open `http://localhost:8000` in your web browser.

---

## 🧪 Testing

We use `pytest` for unit testing. The test suite uses unittest mocking to test all APIs, specialist agents, and geocoding tools without sending network requests or calling the live GenAI APIs.

Run tests:
```bash
python -m pytest
```

---

## 🐳 Containerization & Deployment

### Build Docker Image
```bash
docker build -t gcr.io/your-project-id/wanderai:latest .
```

### Run Docker Locally
```bash
docker run -p 8080:8080 --env-file .env gcr.io/your-project-id/wanderai:latest
```

### Deploy to Google Cloud Run
```bash
gcloud run deploy wanderai \
  --image gcr.io/your-project-id/wanderai:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

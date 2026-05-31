import asyncio
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from config import settings
from tools import geocode_destination, get_us_weather

logger = logging.getLogger("wanderai.agents")

# ====================================================
# PYDANTIC RESPONSE MODELS (Strict Schemas)
# ====================================================

class WeatherResponse(BaseModel):
    temperature: str = Field(..., description="Current temperature and unit (e.g. '72°F' or '22°C')")
    conditions: str = Field(..., description="Weather conditions (e.g. 'Sunny', 'Rainy', 'Snowy')")
    best_time_to_visit: str = Field(..., description="Best season or month to visit this destination")
    travel_tips: str = Field(..., description="Weather-specific travel advice (e.g. 'pack an umbrella', 'wear sunscreen')")

class Activity(BaseModel):
    time: str = Field(..., description="Suggested time range, e.g. '09:00 AM - 11:30 AM'")
    activity: str = Field(..., description="Short title of the activity")
    location: str = Field(..., description="Name of the real physical location")
    description: str = Field(..., description="Detailed description of what to do there")

class DayItinerary(BaseModel):
    day: int = Field(..., description="Day number starting from 1")
    morning: Activity = Field(..., description="Morning activity details")
    afternoon: Activity = Field(..., description="Afternoon activity details")
    evening: Activity = Field(..., description="Evening activity details")
    local_tips: List[str] = Field(..., description="Tips specific to the day's locations/activities")
    transportation: str = Field(..., description="Best way to travel between these locations")

class ItineraryResponse(BaseModel):
    destination: str = Field(..., description="Name of the destination")
    days: List[DayItinerary] = Field(..., description="Daily itineraries")

class Hotel(BaseModel):
    name: str = Field(..., description="Name of the hotel")
    description: str = Field(..., description="Brief description highlighting key amenities/location")
    price_range: str = Field(..., description="Approximate price per night (e.g., '$100 - $150')")
    tier: str = Field(..., description="Category: 'Budget', 'Mid-range', or 'Luxury'")
    rating: float = Field(..., description="Rating (out of 5.0)")

class Restaurant(BaseModel):
    name: str = Field(..., description="Name of the restaurant")
    description: str = Field(..., description="Brief description/vibe of the restaurant")
    must_try_dishes: List[str] = Field(..., description="List of signature or must-try dishes")
    specialty: str = Field(..., description="Type of cuisine or specialty (e.g., 'French Bistro', 'Sushi')")

class HotelFoodResponse(BaseModel):
    hotels: List[Hotel] = Field(..., description="Recommended accommodations across budget tiers")
    restaurants: List[Restaurant] = Field(..., description="Recommended dining options")

class CostBreakdown(BaseModel):
    flights: str = Field(..., description="Estimated flight cost details/range")
    hotels: str = Field(..., description="Estimated total hotel cost details")
    food: str = Field(..., description="Estimated food and dining budget")
    activities: str = Field(..., description="Estimated sightseeing/activities cost")
    local_transport: str = Field(..., description="Estimated local transport cost (taxi, metro, etc.)")
    total: str = Field(..., description="Estimated total cost for the trip")

class BudgetResponse(BaseModel):
    currency: str = Field(..., description="Currency used for calculations (usually USD or local currency)")
    budget_tier: CostBreakdown = Field(..., description="Cost breakdown for a budget trip")
    mid_range_tier: CostBreakdown = Field(..., description="Cost breakdown for a mid-range trip")
    luxury_tier: CostBreakdown = Field(..., description="Cost breakdown for a luxury/premium trip")

class PackingItem(BaseModel):
    item: str = Field(..., description="Name of the item to pack")
    quantity: str = Field(..., description="Suggested quantity (e.g., '3 pairs', '1 jacket', 'as needed')")
    necessity: str = Field(..., description="Priority: 'Required' or 'Optional'")

class PackingResponse(BaseModel):
    clothes: List[PackingItem] = Field(..., description="Clothing and footwear")
    documents: List[PackingItem] = Field(..., description="Passports, visas, reservations, IDs")
    electronics: List[PackingItem] = Field(..., description="Chargers, adapters, devices")
    medicines: List[PackingItem] = Field(..., description="First aid, prescriptions, toiletries")
    accessories: List[PackingItem] = Field(..., description="Bags, umbrellas, sunglasses, and other misc items")

# Orchestrator outputs
class TravelPlan(BaseModel):
    destination: str
    days: int
    budget: str
    interests: List[str]
    travelers: int
    weather: WeatherResponse
    itinerary: ItineraryResponse
    hotels_food: HotelFoodResponse
    budget_details: BudgetResponse
    packing: PackingResponse

class TravelPlanRequest(BaseModel):
    destination: str = Field(..., min_length=1, max_length=100, description="Target destination")
    days: int = Field(..., ge=1, le=30, description="Number of days")
    budget: str = Field(..., description="Budget tier: 'Budget', 'Mid-range', or 'Luxury'")
    interests: List[str] = Field(default_factory=list, description="List of travel interests")
    travelers: int = Field(default=1, ge=1, le=50, description="Number of travelers")

# ====================================================
# BASE AGENT
# ====================================================

class BaseAgent:
    """Base class for all specialist agents."""
    def __init__(self, client: genai.Client):
        self.client = client
        self.model_name = "gemini-2.5-flash"

# ====================================================
# SPECIALIST AGENTS
# ====================================================

class WeatherAgent(BaseAgent):
    """Responsible for fetching and formatting weather details."""
    
    async def generate(self, destination: str) -> WeatherResponse:
        logger.info(f"WeatherAgent starting for: {destination}")
        geo = await geocode_destination(destination)
        
        us_weather_summary = None
        if geo and geo["is_us"]:
            logger.info(f"{destination} is in US. Querying weather.gov...")
            us_weather_summary = await get_us_weather(geo["latitude"], geo["longitude"])
            
        system_instruction = (
            "You are a weather expert. Your job is to return structured weather information "
            "for the requested destination. Be precise, providing realistic temperatures, conditions, "
            "the best time to visit, and useful weather-related travel tips."
        )
        
        prompt = f"Provide a weather overview and details for: {destination}."
        if us_weather_summary:
            prompt += f"\nHere is current forecast data from National Weather Service:\n{us_weather_summary}"
            
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=WeatherResponse,
            system_instruction=system_instruction,
            temperature=0.2
        )
        
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        
        return WeatherResponse.model_validate_json(response.text)


class ItineraryAgent(BaseAgent):
    """Responsible for generating detailed day-by-day itineraries."""
    
    async def generate(
        self, 
        destination: str, 
        days: int, 
        budget: str, 
        interests: List[str], 
        travelers: int
    ) -> ItineraryResponse:
        logger.info(f"ItineraryAgent generating {days}-day plan for {destination} (Budget: {budget})")
        
        interests_str = ", ".join(interests) if interests else "General sightseeing"
        system_instruction = (
            "You are a local tour guide and expert travel planner. Create a highly detailed, "
            "day-by-day itinerary. For each day, plan Morning, Afternoon, and Evening activities. "
            "Use real, active locations and sights. Include times, local secrets/tips, and realistic "
            "transportation suggestions between points."
        )
        
        prompt = (
            f"Generate a detailed {days}-day itinerary for a trip to {destination}.\n"
            f"Travelers: {travelers}\n"
            f"Budget tier: {budget}\n"
            f"Interests: {interests_str}\n"
            f"Please research real tourist locations, landmarks, and timings using search grounding."
        )
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ItineraryResponse,
            system_instruction=system_instruction,
            temperature=0.2
        )
        
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        
        return ItineraryResponse.model_validate_json(response.text)


class HotelAndFoodAgent(BaseAgent):
    """Responsible for finding accommodation and dining recommendations."""
    
    async def generate(self, destination: str) -> HotelFoodResponse:
        logger.info(f"HotelAndFoodAgent searching in: {destination}")
        
        system_instruction = (
            "You are a travel hospitality agent. Recommend real hotels and restaurants. "
            "You must provide exactly 3 hotels (1 Budget, 1 Mid-range, 1 Luxury) with actual "
            "hotel names, price estimates, and ratings. "
            "Recommend at least 3 restaurants with their specialty, local favorites, and must-try dishes."
        )
        
        prompt = (
            f"Find real, popular hotels and restaurants in {destination}.\n"
            f"Research actual options with up-to-date ratings and price ranges."
        )
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=HotelFoodResponse,
            system_instruction=system_instruction,
            temperature=0.2
        )
        
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        
        return HotelFoodResponse.model_validate_json(response.text)


class BudgetAgent(BaseAgent):
    """Responsible for generating realistic cost estimations."""
    
    async def generate(self, destination: str, days: int) -> BudgetResponse:
        logger.info(f"BudgetAgent computing estimates for: {destination} ({days} days)")
        
        system_instruction = (
            "You are a travel finance analyst. Estimate detailed expenses for a trip to the destination. "
            "Provide realistic costs in USD (total and details) for Flights, Hotels, Food, Activities, and "
            "Local Transport across three budget tiers: Budget, Mid-range, and Luxury."
        )
        
        prompt = (
            f"Estimate the trip expenses for a {days}-day stay in {destination}.\n"
            f"Assume a standard trip departure. Provide structured cost ranges for flights, total hotels, "
            f"meals, sightseeing, and local taxis/public transit."
        )
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BudgetResponse,
            system_instruction=system_instruction,
            temperature=0.2
        )
        
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        
        return BudgetResponse.model_validate_json(response.text)


class PackingAgent(BaseAgent):
    """Responsible for compiling packing lists based on travel factors."""
    
    async def generate(self, destination: str, days: int, weather_conditions: str) -> PackingResponse:
        logger.info(f"PackingAgent compiling list for: {destination} ({days} days, Weather: {weather_conditions})")
        
        system_instruction = (
            "You are a professional travel packing specialist. Prepare a complete, itemized packing checklist "
            "customized for the destination, number of days, and predicted weather. Organize items into "
            "Clothes, Documents, Electronics, Medicines, and Accessories with recommended quantities."
        )
        
        prompt = (
            f"Create a packing list for a {days}-day trip to {destination}.\n"
            f"Weather context: {weather_conditions}."
        )
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PackingResponse,
            system_instruction=system_instruction,
            temperature=0.2
        )
        
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        
        return PackingResponse.model_validate_json(response.text)

# ====================================================
# MEMORY SERVICE & ORCHESTRATOR
# ====================================================

class InMemorySessionService:
    """Manages multi-turn conversation histories in memory."""
    def __init__(self):
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}
        # Stores the last generated travel plan for the session to support follow-ups
        self._session_plans: Dict[str, TravelPlan] = {}

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        return self._sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": role, "content": content})

    def get_plan(self, session_id: str) -> Optional[TravelPlan]:
        return self._session_plans.get(session_id)

    def set_plan(self, session_id: str, plan: TravelPlan) -> None:
        self._session_plans[session_id] = plan

    def clear(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id] = []
        if session_id in self._session_plans:
            del self._session_plans[session_id]


class OrchestratorIntent(BaseModel):
    action: str = Field(..., description="Action to take: 'CREATE_PLAN', 'MODIFY_PLAN', or 'CHAT'")
    destination: Optional[str] = Field(None, description="The destination extracted or inferred")
    days: Optional[int] = Field(None, description="Duration in days, default to 5 if not specified but destination is present")
    budget: Optional[str] = Field(None, description="Budget level: 'Budget', 'Mid-range', or 'Luxury'")
    interests: List[str] = Field(default_factory=list, description="Extracted user interests (e.g. ['food', 'museums'])")
    travelers: Optional[int] = Field(1, description="Number of travelers, default to 1")
    modification_instructions: Optional[str] = Field(None, description="Instructions on what to change in the existing plan")
    conversational_reply: Optional[str] = Field(None, description="Direct friendly text reply if the action is CHAT")


class TravelOrchestratorAgent(BaseModel):
    """Master agent that manages the user interaction flow, triggers sub-agents, and holds memory."""
    
    # Pydantic v2 configuration
    model_config = {"arbitrary_types_allowed": True}
    
    client: genai.Client
    weather_agent: WeatherAgent
    itinerary_agent: ItineraryAgent
    hotel_food_agent: HotelAndFoodAgent
    budget_agent: BudgetAgent
    packing_agent: PackingAgent
    session_service: InMemorySessionService

    def __init__(self, client: genai.Client):
        super().__init__(
            client=client,
            weather_agent=WeatherAgent(client),
            itinerary_agent=ItineraryAgent(client),
            hotel_food_agent=HotelAndFoodAgent(client),
            budget_agent=BudgetAgent(client),
            packing_agent=PackingAgent(client),
            session_service=InMemorySessionService()
        )

    async def _determine_intent(self, message: str, history: List[Dict[str, Any]], has_existing_plan: bool) -> OrchestratorIntent:
        """Invokes Gemini to analyze user message, history, and extract intent and parameters."""
        logger.info(f"Determining orchestrator intent for message: '{message}'")
        
        history_formatted = "\n".join(
            [f"{msg['role'].upper()}: {msg['content']}" for msg in history[-6:]]
        )
        
        system_instruction = (
            "You are the routing and NLU layer of WanderAI. Analyze the user's message and session history "
            "to classify the action as:\n"
            "- 'CREATE_PLAN': If they request a new trip or plan to a destination (e.g. 'Plan a trip to Paris').\n"
            "- 'MODIFY_PLAN': If they want to modify an existing travel plan (e.g. 'change day 3', 'make it budget', 'add hiking').\n"
            "- 'CHAT': If it is a general question, greeting, or comment (e.g. 'hi', 'which adapter do I need?', 'thanks').\n"
            "Extract all relevant parameters (destination, days, budget, interests, travelers) as best as you can.\n"
            "If the action is CHAT, compose a friendly conversational reply directly."
        )
        
        prompt = (
            f"Existing Plan in Context: {'Yes' if has_existing_plan else 'No'}\n\n"
            f"Conversation History:\n{history_formatted}\n\n"
            f"User Message: {message}"
        )
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=OrchestratorIntent,
            system_instruction=system_instruction,
            temperature=0.1
        )
        
        response = await self.client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config
        )
        
        return OrchestratorIntent.model_validate_json(response.text)

    async def chat(self, message: str, session_id: str, destination: str = "") -> Dict[str, Any]:
        """Main entry point for user chat interaction."""
        logger.info(f"Orchestrator chat. Session: {session_id}, Destination Override: '{destination}'")
        
        # 1. Update memory with user input
        self.session_service.add_message(session_id, "user", message)
        history = self.session_service.get_history(session_id)
        existing_plan = self.session_service.get_plan(session_id)
        
        # 2. Extract parameters and intent
        intent = await self._determine_intent(message, history, existing_plan is not None)
        logger.info(f"Orchestrated intent: {intent.action} (Destination: {intent.destination})")
        
        # Override destination/details if explicitly passed via REST parameters
        target_destination = destination or intent.destination
        
        if (intent.action == "CREATE_PLAN" or target_destination) and intent.action != "CHAT":
            # Generate a complete plan using specialist agents
            dest = target_destination or "Paris"
            days = intent.days or 5
            budget = intent.budget or "Mid-range"
            interests = intent.interests or ["sightseeing"]
            travelers = intent.travelers or 1
            
            try:
                # Run agents in parallel
                logger.info("Executing sub-agents in parallel...")
                weather_task = self.weather_agent.generate(dest)
                itinerary_task = self.itinerary_agent.generate(dest, days, budget, interests, travelers)
                hotel_food_task = self.hotel_food_agent.generate(dest)
                budget_task = self.budget_agent.generate(dest, days)
                
                # Fetch first 4
                weather, itinerary, hotel_food, budget_details = await asyncio.gather(
                    weather_task, itinerary_task, hotel_food_task, budget_task
                )
                
                # Generate packing list sequentially using weather conditions from the weather task
                packing = await self.packing_agent.generate(dest, days, weather.conditions)
                
                plan = TravelPlan(
                    destination=dest,
                    days=days,
                    budget=budget,
                    interests=interests,
                    travelers=travelers,
                    weather=weather,
                    itinerary=itinerary,
                    hotels_food=hotel_food,
                    budget_details=budget_details,
                    packing=packing
                )
                
                # Store the plan in memory
                self.session_service.set_plan(session_id, plan)
                
                # Generate a summary intro
                summary_prompt = (
                    f"Write a warm, enthusiastic summary introducing the travel plan for {dest} ({days} days, "
                    f"{budget} tier). Briefly highlight the weather ({weather.conditions}), top hotel, "
                    f"and a key day 1 activity from the itinerary. Keep the intro under 4 sentences."
                )
                summary_res = await self.client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=summary_prompt,
                    config=types.GenerateContentConfig(temperature=0.7)
                )
                reply = summary_res.text.strip()
                
                # Update memory with assistant response
                self.session_service.add_message(session_id, "assistant", reply)
                
                return {
                    "reply": reply,
                    "plan": plan.model_dump(),
                    "session_id": session_id
                }
                
            except Exception as e:
                logger.exception(f"Error during parallel sub-agent compilation: {e}")
                err_msg = f"Sorry, I encountered an issue planning your trip to {dest}. Details: {str(e)}"
                self.session_service.add_message(session_id, "assistant", err_msg)
                return {
                    "reply": err_msg,
                    "plan": None,
                    "session_id": session_id
                }
                
        elif intent.action == "MODIFY_PLAN" and existing_plan:
            # Modify the existing plan based on modification instructions
            logger.info("Modifying existing plan...")
            try:
                instruct = intent.modification_instructions or message
                
                # Let's run a quick agent revision call
                # E.g. revise itinerary
                system_instruction = (
                    f"You are an expert itinerary editor. Modify the existing itinerary according to "
                    f"the user's instructions: '{instruct}'. Keep everything else similar but apply "
                    f"the modifications accurately."
                )
                
                prompt = (
                    f"Original Itinerary:\n{existing_plan.itinerary.model_dump_json()}\n\n"
                    f"Instructions: {instruct}"
                )
                
                itinerary_res = await self.client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ItineraryResponse,
                        system_instruction=system_instruction,
                        temperature=0.2
                    )
                )
                
                revised_itinerary = ItineraryResponse.model_validate_json(itinerary_res.text)
                existing_plan.itinerary = revised_itinerary
                
                # Re-generate packing list in case duration or conditions changed
                existing_plan.packing = await self.packing_agent.generate(
                    existing_plan.destination, 
                    existing_plan.days, 
                    existing_plan.weather.conditions
                )
                
                self.session_service.set_plan(session_id, existing_plan)
                
                reply = f"I've updated your itinerary for {existing_plan.destination} based on: '{instruct}'."
                self.session_service.add_message(session_id, "assistant", reply)
                
                return {
                    "reply": reply,
                    "plan": existing_plan.model_dump(),
                    "session_id": session_id
                }
            except Exception as e:
                logger.exception(f"Error modifying plan: {e}")
                err_msg = f"I tried to update your plan but encountered an error: {str(e)}"
                self.session_service.add_message(session_id, "assistant", err_msg)
                return {
                    "reply": err_msg,
                    "plan": existing_plan.model_dump(),
                    "session_id": session_id
                }
        else:
            # Standard conversational interaction (CHAT action)
            reply = intent.conversational_reply or "How can I help you plan your next adventure?"
            self.session_service.add_message(session_id, "assistant", reply)
            
            # Return existing plan in context if there is one
            plan_dump = existing_plan.model_dump() if existing_plan else None
            
            return {
                "reply": reply,
                "plan": plan_dump,
                "session_id": session_id
            }

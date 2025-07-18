import streamlit as st
import requests
import os
import json
import re
from openai import AzureOpenAI
from dotenv import load_dotenv
import time
import uuid

# Load environment variables
load_dotenv()

# API Keys
SERP_API_KEY = os.getenv("SERP_API_KEY")
FRESH_API_KEY = os.getenv("FRESH_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Initialize OpenAI client
try:
    client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
except Exception as e:
    st.error(f"Failed to initialize Azure OpenAI client: {e}")
    client = None

# --- Function to parse recruiter query using GPT ---
def parse_recruiter_query(query):
    """Parse recruiter query using AI to extract structured data"""
    if not client:
        return {"error": "Azure OpenAI client not available"}

    try:
        system_prompt = """You are an AI assistant that extracts structured recruitment information from natural language queries.

        Fields to extract:
        - job_title: ONLY the exact position title they're hiring for (e.g., "Python Developer", "Data Scientist"). 
          DO NOT include phrases like "looking for", "need a", "hiring", etc.
        - skills: Array of required technical skills mentioned (e.g., ["Python", "Django", "SQL"])
        - experience: Required experience in years (numeric value or range). For fresher candidates, use "fresher" exactly.
        - location: Array of city names if multiple cities are mentioned, or single city name as string if only one city is mentioned.
        - work_preference: Work mode preference - one of: "remote", "onsite", "hybrid", null
        - job_type: Employment type - one of: "full-time", "part-time", "contract", "internship", null

        CRITICAL INSTRUCTIONS:
        1. For job_title, NEVER include phrases like "looking for", "need", "hiring", etc.
        2. For experience, if the query mentions "fresher", "fresh graduate", "entry level", use exactly "fresher"
        3. Return ONLY valid JSON without any explanation or additional text.
        4. Use your knowledge to recognize job titles across all industries and domains."""

        user_prompt = f"""Extract recruitment information from this query: "{query}"

        Examples of correct extraction:

        Input: "We are looking for a Python developer with 3 years experience from Mumbai"
        Output: {{"job_title": "Python Developer", "skills": ["Python"], "experience": "3", "location": "Mumbai", "work_preference": null, "job_type": null}}

        Input: "Need a senior React frontend developer with Redux, TypeScript, 5+ years"
        Output: {{"job_title": "React Frontend Developer", "skills": ["React", "Redux", "TypeScript"], "experience": "5+", "location": null, "work_preference": null, "job_type": null}}

        Input: "python developer with 2 year of experience from surat, ahmedabad and mumbai"
        Output: {{"job_title": "Python Developer", "skills": ["Python"], "experience": "2", "location": ["Surat", "Ahmedabad", "Mumbai"], "work_preference": null, "job_type": null}}

        Input: "Remote React developer needed, 5 years experience, Redux, TypeScript"
        Output: {{"job_title": "React Developer", "skills": ["React", "Redux", "TypeScript"], "experience": "5", "location": null, "work_preference": "remote", "job_type": null}}

        Input: "Looking for fresher Java developer from Delhi"
        Output: {{"job_title": "Java Developer", "skills": ["Java"], "experience": "fresher", "location": "Delhi", "work_preference": null, "job_type": null}}

        Now extract from the query: "{query}"

        Remember: 
        1. Extract ONLY the job title without any prefixes like "looking for", "need", etc.
        2. Extract ONLY the city/location name without additional text.
        3. For fresher candidates, use exactly "fresher" as experience value.
        4. Return ONLY valid JSON."""

        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )

        content = response.choices[0].message.content.strip()

        # Clean up JSON if needed
        if content.startswith('```json'):
            content = content[7:-3]
        elif content.startswith('```'):
            content = content[3:-3]

        content = content.strip()

        # Parse JSON
        parsed = json.loads(content)

        # Clean up the result
        cleaned_result = {
            "job_title": parsed.get("job_title", "").strip() if parsed.get("job_title") else None,
            "skills": [skill.strip() for skill in parsed.get("skills", []) if skill.strip()],
            "experience": str(parsed.get("experience", "")).strip() if parsed.get("experience") else None,
            "location": parsed.get("location") if parsed.get("location") else None,
            "work_preference": parsed.get("work_preference"),
            "job_type": parsed.get("job_type"),
            "parsing_status": "success"
        }

        # Apply "my city" detection
        cleaned_result = handle_my_city_phrase(cleaned_result)

        # Ensure all locations are in India
        if cleaned_result.get("location"):
            if isinstance(cleaned_result["location"], list):
                # Filter the list to include only Indian locations
                indian_locations = [loc for loc in cleaned_result["location"] if is_location_in_india(loc)]
                if indian_locations:
                    cleaned_result["location"] = indian_locations
                else:
                    # If no Indian locations found, default to major cities
                    cleaned_result["location"] = MAJOR_INDIAN_CITIES[:3]
                    st.warning("⚠️ No Indian cities detected - defaulting to major Indian cities")
            elif isinstance(cleaned_result["location"], str):
                # Check if single location is in India
                if not is_location_in_india(cleaned_result["location"]):
                    # If not in India, default to major cities
                    cleaned_result["location"] = MAJOR_INDIAN_CITIES[:3]
                    st.warning("⚠️ Non-Indian city detected - defaulting to major Indian cities")

        return cleaned_result

    except json.JSONDecodeError as e:
        st.warning(f"JSON parsing error: {e}")
        return {"error": f"JSON parsing error: {e}"}
    except Exception as e:
        st.warning(f"AI parsing error: {e}")
        return {"error": f"AI parsing error: {e}"}

# Define major Indian cities
MAJOR_INDIAN_CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune", "Kolkata", 
                       "Ahmedabad", "Surat", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", 
                       "Thane", "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Ghaziabad"]

# Known Indian cities for validation
INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune", "Kolkata",
    "Ahmedabad", "Surat", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore",
    "Thane", "Bhopal", "Visakhapatnam", "Pimpri", "Patna", "Vadodara",
    "Ghaziabad", "Ludhiana", "Agra", "Nashik", "Faridabad", "Meerut",
    "Rajkot", "Kalyan", "Vasai", "Varanasi", "Srinagar", "Dhanbad",
    "Jodhpur", "Amritsar", "Raipur", "Allahabad", "Coimbatore", "Jabalpur",
    "Gwalior", "Vijayawada", "Madurai", "Guwahati", "Chandigarh", "Hubli",
    "Mysore", "Tiruchirappalli", "Bareilly", "Aligarh", "Tiruppur", "Gurgaon",
    "Noida", "Kochi", "Kozhikode", "Thiruvananthapuram", "Jammu", "Mangalore",
    "Erode", "Belgaum", "Ambattur", "Tirunelveli", "Malegaon", "Gaya"
]

# Function to check if a location is in India
def is_location_in_india(location_name):
    """Check if a location is likely in India."""
    if not isinstance(location_name, str):
        return False

    location_lower = location_name.lower()

    # Check against major cities
    if location_lower in [city.lower() for city in MAJOR_INDIAN_CITIES]:
        return True

    # Check against all Indian cities
    if location_lower in [city.lower() for city in INDIAN_CITIES]:
        return True

    # A simple check for common non-Indian locations to exclude
    non_indian_cities = ["london", "new york", "san francisco", "dubai", "singapore", "toronto"]
    if location_lower in non_indian_cities:
        return False

    # Assume it might be an Indian city if not explicitly non-Indian
    return True

# Function to detect "in my city" phrases and replace with major Indian cities
def handle_my_city_phrase(parsed_data):
    """Replace 'in my city' with major Indian cities"""
    if not parsed_data.get("location"):
        return parsed_data

    location = parsed_data["location"]

    # Check for "my city" phrases
    my_city_phrases = ["my city", "our city", "my town", "our town", "my area", "our area"]

    if isinstance(location, str):
        location_lower = location.lower()
        if any(phrase in location_lower for phrase in my_city_phrases):
            # Replace with list of major Indian cities
            parsed_data["location"] = MAJOR_INDIAN_CITIES[:5]  # Use top 5 cities
            st.info("📍 'My city' detected - searching across major Indian cities: " + 
                   ", ".join(MAJOR_INDIAN_CITIES[:5]))

    return parsed_data

# --- Function to extract location from profile snippet/title ---
def extract_location_from_text(text):
    """Extract location information from profile text"""
    if not text:
        return "Location not specified"

    # Common location patterns
    location_patterns = [
        r'(?:at|in|from|based in|located in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Area',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*,\s*India',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\|\s*',
        r'\|\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    ]

    # Known Indian cities for validation
    indian_cities = [
        "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune", "Kolkata",
        "Ahmedabad", "Surat", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore",
        "Thane", "Bhopal", "Visakhapatnam", "Pimpri", "Patna", "Vadodara",
        "Ghaziabad", "Ludhiana", "Agra", "Nashik", "Faridabad", "Meerut",
        "Rajkot", "Kalyan", "Vasai", "Varanasi", "Srinagar", "Dhanbad",
        "Jodhpur", "Amritsar", "Raipur", "Allahabad", "Coimbatore", "Jabalpur",
        "Gwalior", "Vijayawada", "Madurai", "Guwahati", "Chandigarh", "Hubli",
        "Mysore", "Tiruchirappalli", "Bareilly", "Aligarh", "Tiruppur", "Gurgaon",
        "Noida", "Kochi", "Kozhikode", "Thiruvananthapuram", "Jammu", "Mangalore",
        "Erode", "Belgaum", "Ambattur", "Tirunelveli", "Malegaon", "Gaya"
    ]

    for pattern in location_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Check if the matched text is a known Indian city
            if match.title() in indian_cities:
                return match.title()

    # If no pattern matches, try to find any Indian city mentioned
    text_lower = text.lower()
    for city in indian_cities:
        if city.lower() in text_lower:
            return city

    return "Location not clearly specified"

# --- Pagination Management Functions ---
def initialize_pagination():
    """Initialize pagination session state"""
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    if "results_per_page" not in st.session_state:
        st.session_state.results_per_page = 10
    if "total_pages" not in st.session_state:
        st.session_state.total_pages = 1
    if "search_cache" not in st.session_state:
        st.session_state.search_cache = {}
    if "pagination_key" not in st.session_state:
        st.session_state.pagination_key = "default"

def handle_pagination_click(action):
    """Handle pagination button clicks"""
    if action == "next":
        st.session_state.current_page += 1
    elif action == "prev" and st.session_state.current_page > 1:
        st.session_state.current_page -= 1
    elif isinstance(action, int) and action > 0:
        st.session_state.current_page = action

def get_pagination_controls():
    """Display pagination controls and return current page settings"""
    initialize_pagination()

    # Display cache status
    if 'search_cache' in st.session_state:
        cached_pages = len(st.session_state.search_cache)
        if cached_pages > 0:
            st.info(f"💾 {cached_pages} pages cached for faster navigation")

    st.markdown("### 📄 Pagination Controls")
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    with col1:
        # Previous button with callback
        prev_disabled = st.session_state.current_page <= 1
        if st.button("⬅️ Previous", key="prev_btn_fixed", disabled=prev_disabled, on_click=handle_pagination_click, args=("prev",)):
            # The actual navigation happens in the callback
            pass

    with col2:
        # Page info
        st.markdown(f"**Page {st.session_state.current_page}**")

    with col3:
        # Next button with callback
        if st.button("Next ➡️", key="next_btn_fixed", on_click=handle_pagination_click, args=("next",)):
            # The actual navigation happens in the callback
            pass

    with col4:
        # Results per page selector
        options = [5, 10, 15, 20, 25]
        index = options.index(st.session_state.results_per_page) if st.session_state.results_per_page in options else 1
        
        def on_change_results():
            # Reset to page 1 when changing results per page
            st.session_state.current_page = 1
            # Clear cache when changing results per page
            st.session_state.search_cache = {}
        
        st.selectbox(
            "Results per page:", 
            options,
            index=index,
            key="results_per_page_selector",
            on_change=on_change_results
        )
        # Update session state after widget
        st.session_state.results_per_page = st.session_state.results_per_page_selector

    with col5:
        # Go to page input with callback
        def go_to_page():
            if st.session_state.go_to_page_input >= 1:
                handle_pagination_click(st.session_state.go_to_page_input)
        
        st.number_input(
            "Go to page:", 
            min_value=1, 
            max_value=100,
            value=st.session_state.current_page,
            step=1,
            key="go_to_page_input",
            on_change=go_to_page
        )

    return st.session_state.current_page, st.session_state.results_per_page

def reset_pagination():
    """Reset pagination to first page and clear cache"""
    st.session_state.current_page = 1
    # Clear cache when starting new search
    if 'search_cache' in st.session_state:
        st.session_state.search_cache = {}
    # Generate new pagination key to force refresh
    st.session_state.pagination_key = str(uuid.uuid4())

def search_with_pagination(parsed_data):
    """Main function to perform search with pagination controls"""
    # Initialize pagination
    initialize_pagination()

    # Get current page and results per page
    current_page, results_per_page = get_pagination_controls()

    # Create cache key based on search query and pagination
    search_query_str = str(parsed_data)  # Convert parsed data to string for cache key
    cache_key = f"{hash(search_query_str)}_page_{current_page}_size_{results_per_page}"

    # Check if we have cached results
    if cache_key in st.session_state.search_cache:
        search_results = st.session_state.search_cache[cache_key]
        st.info(f"📋 Loaded from cache (Page {current_page})")
    else:
        with st.spinner(f"🔍 Searching... (Page {current_page}, {results_per_page} results per page)"):
            search_results = get_linkedin_urls(
                parsed_data,
                page_number=current_page,
                results_per_page=results_per_page
            )
            # Cache the results
            st.session_state.search_cache[cache_key] = search_results
            st.success(f"✅ Fetched and cached (Page {current_page})")

    # Show debug info
    st.write(f"🔄 Showing page: {current_page}, Results per page: {results_per_page}")

    # Get fetch_detailed setting
    fetch_detailed = st.session_state.get('fetch_detailed', True)

    # Process and display results
    if search_results["urls"] and fetch_detailed:
        with st.spinner("📊 Analyzing candidate profiles..."):
            candidates = []
            linkedin_urls = search_results["urls"]
            location_mismatch_profiles = search_results.get("location_mismatch_profiles", [])

            for i, url_data in enumerate(linkedin_urls):
                profile_data = fetch_profile_data(url_data["url"])
                if profile_data and "data" in profile_data:
                    if matches_location_criteria(profile_data, parsed_data):
                        score = score_profile(profile_data, url_data, parsed_data)
                        candidate = {
                            "url": url_data["url"],
                            "profile_data": profile_data["data"],
                            "search_metadata": url_data,
                            "score": score,
                            "match_category": get_match_category(score)
                        }
                        candidates.append(candidate)
                    else:
                        profile_location = profile_data["data"].get('location', 'Not specified')
                        location_mismatch_profiles.append({
                            "url": url_data["url"],
                            "name": profile_data["data"].get('full_name', 'Unknown'),
                            "target_location": parsed_data.get("location", ""),
                            "actual_location": profile_location
                        })
                else:
                    candidates.append({
                        "url": url_data["url"],
                        "profile_data": None,
                        "search_metadata": url_data,
                        "score": 0,
                        "match_category": "❓ Unable to analyze"
                    })

                time.sleep(0.2)

            candidates.sort(key=lambda x: x["score"], reverse=True)
            st.session_state.search_results = candidates

            display_detailed_candidates(candidates, location_mismatch_profiles)

    elif search_results["urls"]:
        st.success(f"✅ Found {len(search_results['urls'])} profiles")

        for idx, profile in enumerate(search_results["urls"], 1):
            with st.expander(f"Profile {idx}: {profile['name']}", expanded=False):
                st.write(f"**URL:** {profile['url']}")
                st.write(f"**Title:** {profile['title']}")
                st.write(f"**Location:** {profile['extracted_location']}")
                st.write(f"**Snippet:** {profile['snippet'][:200]}...")
                if profile.get('matched_city'):
                    st.write(f"**Matched City:** {profile['matched_city']}")
    else:
        st.warning("❌ No LinkedIn profiles found on this page")

    # Footer pagination info
    if search_results.get("search_info"):
        info = search_results["search_info"]
        with st.container():
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Page", info.get('current_page', 1))
            with col2:
                st.metric("Results Per Page", info.get('results_per_page', 10))
            with col3:
                st.metric("Results Found", len(search_results["urls"]))

    return search_results


# --- Function to get LinkedIn URLs using SERP API ---
def get_linkedin_urls(parsed_data, page_number=1, results_per_page=10):
    job_title = parsed_data.get("job_title", "")
    location = parsed_data.get("location", "")
    skills = parsed_data.get("skills", [])
    experience = parsed_data.get("experience")
    work_preference = parsed_data.get("work_preference")

    all_linkedin_urls = []
    all_ignored_profiles = []
    all_location_mismatch_profiles = []

    # Handle multiple cities
    cities_to_search = []
    if location:
        if isinstance(location, list):
            cities_to_search = location
        else:
            cities_to_search = [location]
    else:
        # Default to top 3 major cities if no location specified
        cities_to_search = MAJOR_INDIAN_CITIES[:3]

    # Limit to a reasonable number of cities to avoid too many API calls
    if len(cities_to_search) > 5:
        st.warning(f"⚠️ Limiting search to top 5 cities out of {len(cities_to_search)} detected")
        cities_to_search = cities_to_search[:5]

    # Display cities being searched
    city_list_str = ", ".join(cities_to_search)
    st.info(f"📍 Searching across {len(cities_to_search)} cities: {city_list_str}")

    # Build a SINGLE search query for all cities
    search_query = 'site:linkedin.com/in '

    # Add job title (required)
    if job_title:
        search_query += f'"{job_title}" '

    # Add all cities with OR operator
    if len(cities_to_search) > 1:
        city_query = " OR ".join([f'"{city}"' for city in cities_to_search])
        search_query += f'({city_query}) '
    else:
        search_query += f'"{cities_to_search[0]}" '

    # Add "India" to ensure Indian candidates only
    search_query += '"India" '

    # Add experience handling - FIX FOR FRESHER ISSUE
    if experience:
        if experience.lower() == "fresher":
            search_query += '"fresher" OR "entry level" OR "recent graduate" OR "fresh graduate" '
        else:
            # Handle experience ranges like "2-4", "5+" etc.
            exp_clean = experience.replace('+', '').replace('-', ' ').split()[0]
            search_query += f'"{exp_clean} years" OR "{exp_clean}+ years" '

    # Add most important skill only (for better results)
    if skills and len(skills) > 0:
        search_query += f'"{skills[0]}" '

    # Add work preference if specified
    if work_preference:
        search_query += f'"{work_preference}" '

    # Exclude job posting keywords
    search_query += '-"job" -"jobs" -"hiring" -"vacancy" -"openings" -"career" -"apply"'

    # Display the search query for debugging
    st.code(search_query, language="text")

    # Calculate start parameter for pagination
    start_index = (page_number - 1) * results_per_page

    # SERP API parameters with pagination
    params = {
        "engine": "google",
        "q": search_query.strip(),
        "api_key": os.getenv("SERP_API_KEY"),
        "hl": "en",
        "gl": "in",
        "google_domain": "google.co.in",
        "location": "India",
        "num": results_per_page,
        "start": start_index,
        "safe": "active"
    }

    try:
        response = requests.get("https://serpapi.com/search", params=params)

        if response.status_code == 200:
            data = response.json()
            organic_results = data.get("organic_results", [])

            linkedin_urls = []
            ignored_profiles = []
            location_mismatch_profiles = []

            for result in organic_results:
                title = result.get("title", "")
                link = result.get("link", "")
                snippet = result.get("snippet", "")

                # Check if it's a LinkedIn profile URL
                if ("linkedin.com/in/" in link or "in.linkedin.com/in/" in link):
                    # Extract name from title (LinkedIn titles usually contain the person's name)
                    name_parts = title.split(' - ')[0].split(' | ')[0].split(' at ')[0]
                    if len(name_parts) > 50:  # If too long, likely not just a name
                        name = "Professional Profile"
                    else:
                        name = name_parts.strip()

                    # Clean LinkedIn URL
                    clean_url = link.split("?")[0]  # Remove query parameters
                    if "in.linkedin.com" in clean_url:
                        clean_url = clean_url.replace("in.linkedin.com", "linkedin.com")

                    # Extract actual location from profile text
                    profile_location = extract_location_from_text(f"{title} {snippet}")

                    # Check if profile matches any of the target locations
                    location_match = False
                    matched_city = None
                    for city in cities_to_search:
                        if city.lower() in snippet.lower() or city.lower() in title.lower():
                            location_match = True
                            matched_city = city
                            break

                    # Check for job posting keywords
                    job_keywords = ["job", "jobs", "hiring", "vacancy", "openings", "career", "apply", "position"]
                    has_job_keywords = any(keyword in title.lower() or keyword in snippet.lower() for keyword in job_keywords)

                    # Check if profile is from India
                    is_from_india = "india" in snippet.lower() or "india" in title.lower()
                    for city in INDIAN_CITIES:
                        if city.lower() in snippet.lower() or city.lower() in title.lower():
                            is_from_india = True
                            break

                    if not has_job_keywords:
                        if is_from_india:
                            # Add result with metadata
                            linkedin_urls.append({
                                "url": clean_url,
                                "title": title,
                                "name": name,
                                "snippet": snippet,
                                "position": result.get("position", 0),
                                "location_match": location_match,
                                "extracted_location": profile_location,
                                "is_from_india": is_from_india,
                                "matched_city": matched_city  # Store which city matched
                            })

                            # Add to location mismatch list if applicable
                            if not location_match:
                                location_mismatch_profiles.append({
                                    "url": clean_url,
                                    "name": name,
                                    "target_locations": cities_to_search,
                                    "actual_location": profile_location
                                })
                        else:
                            # Add to ignored profiles - not from India
                            ignored_profiles.append({
                                "url": clean_url,
                                "name": name,
                                "reason": "Not from India",
                                "extracted_location": profile_location
                            })
                    else:
                        # Add to ignored profiles - job posting
                        ignored_profiles.append({
                            "url": clean_url,
                            "name": name,
                            "reason": "Contains job posting keywords",
                            "extracted_location": profile_location
                        })

            # Return results
            return {
                "urls": linkedin_urls,
                "total_results": len(linkedin_urls),
                "ignored_profiles": ignored_profiles,
                "location_mismatch_profiles": location_mismatch_profiles,
                "cities_searched": cities_to_search,
                "search_info": {
                    "cities_searched": len(cities_to_search),
                    "current_page": page_number,
                    "results_per_page": results_per_page,
                    "start_index": start_index
                }
            }

        else:
            st.error(f"SERP API Error: {response.status_code}: {response.text}")
            return {
                "urls": [],
                "total_results": 0,
                "ignored_profiles": [],
                "location_mismatch_profiles": [],
                "cities_searched": cities_to_search,
                "search_info": {"error": f"API Error: {response.status_code}"}
            }

    except Exception as e:
        st.error(f"Error fetching LinkedIn URLs: {e}")
        return {
            "urls": [],
            "total_results": 0,
            "ignored_profiles": [],
            "location_mismatch_profiles": [],
            "cities_searched": cities_to_search,
            "search_info": {"error": f"Exception: {str(e)}"}
        }

# --- Function to fetch detailed profile data ---
def fetch_profile_data(linkedin_url):
    """Fetch detailed profile data using Fresh API"""
    url = "https://fresh-linkedin-profile-data.p.rapidapi.com/get-profile-public-data"
    querystring = {
        "linkedin_url": linkedin_url,
        "include_skills": "true",
        "include_certifications": "false"
    }

    headers = {
        "x-rapidapi-key": FRESH_API_KEY,
        "x-rapidapi-host": "fresh-linkedin-profile-data.p.rapidapi.com"
    }

    try:
        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.warning(f"❌ Profile fetch error: {e}")
        return None

# --- Function to check if profile matches location criteria ---
def matches_location_criteria(profile_data, parsed_data):
    """Check if profile matches location criteria with India-only enforcement"""
    if not profile_data or "data" not in profile_data:
        return False

    data = profile_data["data"]
    location_text = data.get('location', '').lower()

    if not location_text:
        return False  # No location in profile

    # First check: Must be in India
    india_keywords = ["india", "bharat", "indian"]
    is_in_india = any(keyword in location_text for keyword in india_keywords)

    # If not explicitly mentioned India, check for Indian cities
    if not is_in_india:
        for city in INDIAN_CITIES:
            if city.lower() in location_text:
                is_in_india = True
                break

    if not is_in_india:
        return False  # Not in India

    # If no specific location criteria, any Indian location is fine
    if not parsed_data.get("location"):
        return True

    # Check if any of the specified locations match
    if isinstance(parsed_data["location"], list):
        for city in parsed_data["location"]:
            if city.lower().strip() in location_text:
                return True
    else:
        if parsed_data["location"].lower().strip() in location_text:
            return True

    return False

# --- Function to score profile match ---
def score_profile(profile_data, search_metadata, parsed_data):
    """Score profile match with enhanced location handling and India-only enforcement"""
    if not profile_data or "data" not in profile_data:
        return 0

    data = profile_data["data"]
    score = 0

    # Get text content for matching
    profile_text = f"{data.get('full_name', '')} {data.get('headline', '')} {data.get('summary', '')} {data.get('skills', '')}"
    metadata_text = f"{search_metadata.get('title', '')} {search_metadata.get('snippet', '')}"

    # First check: Must be in India (critical requirement)
    location_text = data.get('location', '').lower()
    india_keywords = ["india", "bharat", "indian"]
    is_in_india = any(keyword in location_text for keyword in india_keywords)

    # If not explicitly mentioned India, check for Indian cities
    if not is_in_india:
        for city in INDIAN_CITIES:
            if city.lower() in location_text:
                is_in_india = True
                break

    if not is_in_india:
        return 0  # Not in India, zero score

    # Score based on location match (FIXED: only bonus points, not exclusion)
    if parsed_data.get("location"):
        location_matched = False

        if isinstance(parsed_data["location"], list):
            for city in parsed_data["location"]:
                if city.lower().strip() in location_text:
                    location_matched = True
                    score += 8  # Higher bonus for location match
                    break
        elif parsed_data["location"].lower().strip() in location_text:
            location_matched = True
            score += 8

    # Score based on job title match
    if parsed_data.get("job_title"):
        job_title = parsed_data["job_title"].lower()
        if job_title in profile_text:
            score += 6
        # Check headline specifically
        headline = data.get('headline', '').lower()
        if job_title in headline:
            score += 4
        # Partial match
        title_words = job_title.split()
        for word in title_words:
            if len(word) > 2 and word in profile_text:
                score += 2

    # Score based on skills match
    if parsed_data.get("skills"):
        skills_text = data.get('skills', '').lower()
        for skill in parsed_data["skills"]:
            if skill.lower() in skills_text or skill.lower() in profile_text:
                score += 4

    # Score based on experience match (FIXED: proper fresher handling)
    if parsed_data.get("experience"):
        exp_str = str(parsed_data["experience"]).lower()
        if exp_str == "fresher":
            # Look for fresher indicators in profile
            fresher_keywords = ["fresher", "entry level", "recent graduate", "entry-level", "fresh graduate", "new graduate"]
            if any(keyword in profile_text.lower() for keyword in fresher_keywords):
                score += 6
            # Also check for very recent graduation or short experience
            experiences = data.get('experiences', [])
            if len(experiences) == 0 or (len(experiences) == 1 and any(month in experiences[0].get('duration', '').lower() for month in ['month', 'months'])):
                score += 4
        else:
            # Try to extract years from experiences
            experiences = data.get('experiences', [])
            total_years = 0
            for exp in experiences:
                duration = exp.get('duration', '')
                # Simple extraction of years from duration
                years_match = re.findall(r'(\d+)\s*(?:yr|year)', duration.lower())
                if years_match:
                    total_years += int(years_match[0])

            if total_years > 0:
                try:
                    required_exp = int(exp_str.replace('+', '').replace('-', '').split()[0])
                    if total_years >= required_exp:
                        score += 5
                    elif abs(total_years - required_exp) <= 1:  # Close match
                        score += 3
                except:
                    pass

    # Score based on work preference
    if parsed_data.get("work_preference"):
        work_pref = parsed_data["work_preference"].lower()
        if work_pref in profile_text or work_pref in metadata_text:
            score += 3

    return score

# --- Function to get match category ---
def get_match_category(score):
    if score >= 25:
        return "🔥 Excellent Match"
    elif score >= 18:
        return "✅ Good Match"
    elif score >= 12:
        return "⚡ Fair Match"
    else:
        return "📋 Basic Match"

# --- Main Streamlit App ---
st.set_page_config(page_title="AI Recruiter - Enhanced", page_icon="🤝")

st.title("🤖 Enhanced Recruitment AI")
st.markdown("*Find the perfect candidates using natural language queries with detailed LinkedIn profiles*")

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'parsed_data' not in st.session_state:
    st.session_state.parsed_data = {}
if 'current_search' not in st.session_state:
    st.session_state.current_search = ""



# Search interface
recruiter_query = st.text_area(
    "🎯 Describe your ideal candidate:",
    placeholder="e.g., Looking for fresher Python developer from Mumbai OR Data Scientist with 2 years of experience who have knowledge of Python, Machine Learning, Statistics, SQL, AWS",
    height=100
)

# Search configuration
col1, col2 = st.columns(2)
with col1:
    max_profiles = st.selectbox("Max profiles to analyze", [5, 10, 15, 20], index=1)
with col2:
    fetch_detailed = st.checkbox("Fetch detailed profiles", value=True, help="Get full profile data including skills and experience")
    # Store in session state for pagination function
    st.session_state.fetch_detailed = fetch_detailed

# Helper function to display detailed candidate analysis
def display_detailed_candidates(candidates, location_mismatch_profiles):
    """Display detailed candidate analysis results"""
    if candidates:
        # Summary statistics
        total_candidates = len(candidates)
        excellent_matches = sum(1 for c in candidates if c["score"] >= 25)
        good_matches = sum(1 for c in candidates if 18 <= c["score"] < 25)

        st.success(f"🎉 Found {total_candidates} candidate profiles!")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Candidates", total_candidates)
        with col2:
            st.metric("Excellent Matches", excellent_matches)
        with col3:
            st.metric("Good Matches", good_matches)
        with col4:
            avg_score = sum(c["score"] for c in candidates) / len(candidates)
            st.metric("Avg Score", f"{avg_score:.1f}")

    # Display location mismatch profiles with enhanced formatting
    if location_mismatch_profiles:
        st.warning(f"⚠️ **{len(location_mismatch_profiles)} profiles found but location mismatch:**")

        # Group profiles by their actual location
        location_groups = {}
        for profile in location_mismatch_profiles:
            actual_loc = profile.get('actual_location', 'Unknown')
            target_loc = profile.get('target_locations', profile.get('target_location', 'Unknown location'))

            # Create a consistent group key
            if isinstance(target_loc, list):
                group_key = f"Not from {', '.join(target_loc)}"
            else:
                group_key = f"Not from {target_loc}"

            if group_key not in location_groups:
                location_groups[group_key] = []
            location_groups[group_key].append(profile)

        # Display grouped by location with better formatting
        for location_key, profiles in location_groups.items():
            with st.expander(f"{location_key} ({len(profiles)} profiles)"):
                for profile in profiles:
                    st.markdown(f"• **{profile['name']}** - [View Profile]({profile['url']}) - *Found in: {profile['actual_location']}*")

# Search button
if st.button("🔍 Find Candidates", type="primary", use_container_width=True, key="main_search_btn"):
    if recruiter_query.strip():
        # Reset pagination when starting new search
        reset_pagination()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Parse query
        status_text.text("🤖 Understanding your requirements...")
        progress_bar.progress(0.1)

        parsed_data = parse_recruiter_query(recruiter_query)
        st.session_state.parsed_data = parsed_data

        if parsed_data.get("error"):
            st.error(f"❌ Error parsing query: {parsed_data['error']}")
            st.stop()

        # Step 2: Display parsed requirements
        status_text.text("📋 Analyzing your requirements...")
        progress_bar.progress(0.3)
        
        # Display parsed data
        # ... (your existing code)

        # Step 3: Use pagination search function
        status_text.text("🔍 Searching LinkedIn profiles with pagination...")
        progress_bar.progress(0.8)

        # Store the search query to maintain state between reruns
        st.session_state.last_search_query = recruiter_query
        
        # Use the pagination search function
        search_results = search_with_pagination(parsed_data)

        # Clear progress indicators
        progress_bar.progress(1.0)
        status_text.text("✅ Search complete!")
        
        # Small delay to show completion
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
    else:
        st.error("Please enter a search query")

# Show pagination controls for existing search results
elif 'parsed_data' in st.session_state:
    # Only show pagination if we have a previous search
    pass

# Display search results
if st.session_state.search_results:
    st.header("🎯 Candidate Analysis Results")

    # Summary statistics
    total_candidates = len(st.session_state.search_results)
    excellent_matches = sum(1 for c in st.session_state.search_results if c["score"] >= 25)
    good_matches = sum(1 for c in st.session_state.search_results if 18 <= c["score"] < 25)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Candidates", total_candidates)
    with col2:
        st.metric("Excellent Matches", excellent_matches)
    with col3:
        st.metric("Good Matches", good_matches)
    with col4:
        avg_score = sum(c["score"] for c in st.session_state.search_results) / len(st.session_state.search_results)
        st.metric("Avg Score", f"{avg_score:.1f}")

    # Display each candidate
    for idx, candidate in enumerate(st.session_state.search_results):
        with st.expander(f"👤 Candidate {idx + 1} - {candidate['match_category']} (Score: {candidate['score']})", expanded=idx < 3):

            profile_data = candidate.get("profile_data")
            search_metadata = candidate.get("search_metadata", {})

            if profile_data:
                # Detailed profile view
                col1, col2 = st.columns([1, 2])

                with col1:
                    # Profile image
                    if profile_data.get("profile_image_url"):
                        st.image(profile_data["profile_image_url"], width=150)
                    else:
                        st.info("📷 No profile image")

                    # Quick info
                    st.markdown(f"**📍 Location:** {profile_data.get('location', 'Not specified')}")
                    st.markdown(f"**🏢 Current Role:** {profile_data.get('headline', 'Not specified')}")

                    # LinkedIn URL
                    st.markdown(f"**🔗 LinkedIn:** [View Profile]({candidate['url']})")

                with col2:
                    # Name and headline
                    st.markdown(f"### {profile_data.get('full_name', 'Unknown Name')}")
                    st.markdown(f"*{profile_data.get('headline', 'No headline available')}*")

                    # Summary
                    if profile_data.get("summary"):
                        st.markdown("**📝 Summary:**")
                        summary_preview = profile_data["summary"][:300] + "..." if len(profile_data["summary"]) > 300 else profile_data["summary"]
                        st.markdown(summary_preview)

                    # Skills
                    if profile_data.get("skills"):
                        st.markdown("**🛠️ Skills:**")
                        skills = profile_data["skills"]
                        if isinstance(skills, str):
                            skills_list = [s.strip() for s in skills.split(',')][:10]  # Show top 10
                        else:
                            skills_list = skills[:10] if isinstance(skills, list) else []

                        if skills_list:
                            skills_text = " • ".join(skills_list)
                            st.markdown(f"*{skills_text}*")

                    # Experience
                    if profile_data.get("experiences"):
                        st.markdown("**💼 Recent Experience:**")
                        for exp in profile_data["experiences"][:2]:  # Show top 2 experiences
                            company = exp.get("company", "Unknown Company")
                            title = exp.get("title", "Unknown Title")
                            duration = exp.get("duration", "Duration not specified")
                            st.markdown(f"• **{title}** at **{company}** ({duration})")

                    # Education
                    if profile_data.get("education"):
                        st.markdown("**🎓 Education:**")
                        for edu in profile_data["education"][:2]:  # Show top 2 education
                            school = edu.get("school", "Unknown School")
                            degree = edu.get("degree", "Unknown Degree")
                            st.markdown(f"• **{degree}** from **{school}**")
    
                # Match analysis
                st.markdown("---")
                st.markdown("**🎯 Match Analysis:**")

                # Create match reasons
                match_reasons = []
                parsed_data = st.session_state.parsed_data

                if parsed_data.get("job_title"):
                    job_title = parsed_data["job_title"].lower()
                    profile_text = f"{profile_data.get('full_name', '')} {profile_data.get('headline', '')}".lower()
                    if job_title in profile_text:
                        match_reasons.append(f"✅ Job title '{parsed_data['job_title']}' matches profile")

                if parsed_data.get("location"):
                    location_text = profile_data.get('location', '').lower()
                    target_locations = parsed_data["location"] if isinstance(parsed_data["location"], list) else [parsed_data["location"]]
                    for loc in target_locations:
                        if loc.lower() in location_text:
                            match_reasons.append(f"✅ Location matches: {loc}")
                            break

                if parsed_data.get("skills"):
                    skills_text = str(profile_data.get('skills', '')).lower()
                    matched_skills = []
                    for skill in parsed_data["skills"]:
                        if skill.lower() in skills_text:
                            matched_skills.append(skill)
                    if matched_skills:
                        match_reasons.append(f"✅ Skills match: {', '.join(matched_skills)}")

                if match_reasons:
                    for reason in match_reasons:
                        st.markdown(reason)
                else:
                    st.markdown("ℹ️ Basic profile information available")

            else:
                # Basic profile view (when detailed data not available)
                st.markdown(f"**👤 Profile:** {search_metadata.get('name', 'Unknown')}")
                st.markdown(f"**📍 Location:** {search_metadata.get('extracted_location', 'Location not specified')}")
                st.markdown(f"**🔗 LinkedIn:** [View Profile]({candidate['url']})")

                if search_metadata.get("snippet"):
                    st.markdown(f"**📝 Preview:** {search_metadata['snippet']}")

                st.info("💡 Enable 'Fetch detailed profiles' for more information")

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ using Streamlit, OpenAI, and LinkedIn APIs*")

# Add some CSS for better styling
st.markdown("""
<style>
.stExpander > div:first-child {
    background-color: #f0f2f6;
}
.metric-container {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)

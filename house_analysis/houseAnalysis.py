import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
import requests
import webbrowser

# === USER INPUT ===
address = input("Enter an Ottawa address to analyze (e.g., 249 Kimpton Drive, Ottawa): ")
budget = float(input("Enter your target budget (e.g., 650000): "))

# === GEOCODING ===
geolocator = Nominatim(user_agent="smart_property_locator")
location = geolocator.geocode(address)
if not location:
    raise ValueError("Could not locate the address.")

lat, lon = location.latitude, location.longitude
print(f"\nAnalyzing: {location.address} | Coordinates: {lat}, {lon}\n")

# === BASE MAP ===
m = folium.Map(location=[lat, lon], zoom_start=15)
folium.Marker([lat, lon], popup="Selected Property", icon=folium.Icon(color='blue')).add_to(m)

# === WEATHER ===
weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
weather = requests.get(weather_url).json().get("current_weather", {})
if weather:
    folium.Marker(
        [lat + 0.001, lon],
        popup=f"Weather: {weather['temperature']}°C, Wind: {weather['windspeed']} km/h",
        icon=folium.Icon(color='green', icon='cloud')
    ).add_to(m)

# === COMMUTE TO DOWNTOWN OTTAWA ===
downtown = (45.4215, -75.6972)
route_url = f"http://router.project-osrm.org/route/v1/driving/{lon},{lat};{downtown[1]},{downtown[0]}?overview=false"
route_data = requests.get(route_url).json()
commute_time = 0
if route_data.get("routes"):
    commute_time = round(route_data["routes"][0]["duration"] / 60, 1)
    folium.Marker(
        [lat + 0.002, lon],
        popup=f"Commute to Downtown: {commute_time} min",
        icon=folium.Icon(color='purple')
    ).add_to(m)

# === NASA EARTH LIVE VIEW ===
nasa_embed = """
<iframe width="300" height="200" src="https://www.youtube.com/embed/86YLFOog4GM"
frameborder="0" allowfullscreen></iframe>
"""
folium.Marker(
    [lat - 0.001, lon],
    popup=folium.Popup(nasa_embed, max_width=300),
    icon=folium.Icon(color='red', icon='info-sign')
).add_to(m)

# === BUS STOPS (Overpass API) ===
bus_query = f"""
[out:json];
node(around:800,{lat},{lon})[highway=bus_stop];
out;
"""
bus_data = requests.post("http://overpass-api.de/api/interpreter", data={"data": bus_query}).json()
bus_cluster = MarkerCluster(name='Bus Stops').add_to(m)
for node in bus_data["elements"]:
    folium.Marker(
        [node["lat"], node["lon"]],
        popup=node.get("tags", {}).get("name", "Bus Stop"),
        icon=folium.Icon(color='orange', icon='bus')
    ).add_to(bus_cluster)

# === TRAIN STATIONS ===
train_query = f"""
[out:json];
node(around:1500,{lat},{lon})[railway=station];
out;
"""
train_data = requests.post("http://overpass-api.de/api/interpreter", data={"data": train_query}).json()
for node in train_data["elements"]:
    folium.Marker(
        [node["lat"], node["lon"]],
        popup=node.get("tags", {}).get("name", "Train Station"),
        icon=folium.Icon(color='darkred', icon='train')
    ).add_to(m)

# === PARKS / GREEN ZONES ===
parks_query = f"""
[out:json];
node(around:1200,{lat},{lon})[leisure=park];
out;
"""
parks_data = requests.post("http://overpass-api.de/api/interpreter", data={"data": parks_query}).json()
for node in parks_data["elements"]:
    folium.Marker(
        [node["lat"], node["lon"]],
        popup=node.get("tags", {}).get("name", "Park"),
        icon=folium.Icon(color='lightgreen', icon='tree')
    ).add_to(m)

# === SIMULATED METRICS ===
estimated_price = 685000  # Replace this with Realtor.ca or MLS API
school_rating = 8.2
crime_risk = "Low"
walk_score = 78  # Simulated walkability score out of 100

if estimated_price <= budget * 0.9:
    affordability = "Excellent"
elif estimated_price <= budget * 1.05:
    affordability = "Fair"
else:
    affordability = "Expensive"

# === SUMMARY CARD ===
summary = f"""
<b>Affordability:</b> {affordability}<br>
<b>Estimated Price:</b> ${estimated_price:,}<br>
<b>School Rating:</b> {school_rating}/10<br>
<b>Walk Score:</b> {walk_score}/100<br>
<b>Commute Time:</b> {commute_time} min<br>
<b>Crime Risk:</b> {crime_risk}
"""
folium.Marker(
    [lat - 0.002, lon],
    popup=folium.Popup(summary, max_width=300),
    icon=folium.Icon(color='cadetblue', icon='info-sign')
).add_to(m)

# === EXPORT MAP ===
output_file = "smart_property_dashboard.html"
m.save(output_file)
webbrowser.open(output_file)

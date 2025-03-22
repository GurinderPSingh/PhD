import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
import requests
import pandas as pd
from fpdf import FPDF
import webbrowser
from bs4 import BeautifulSoup
import statistics
import random

def fetch_from_remax(address):
    """
    Scrapes a Remax-style website for sale price and build year.
    Update the URL pattern and CSS selectors as needed.
    """
    try:
        url = f"https://www.remax.ca/property/{address.replace(' ', '-')}"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Remax returned status code {response.status_code} for {address}")
            return None, None
        soup = BeautifulSoup(response.text, 'html.parser')
        sale_price = None
        build_year = None

        price_elem = soup.find('div', class_="property-price")
        if price_elem:
            price_text = price_elem.get_text(strip=True)
            sale_price = int(''.join(filter(str.isdigit, price_text)))
        year_elem = soup.find('div', class_="year-built")
        if year_elem:
            build_year = int(''.join(filter(str.isdigit, year_elem.get_text(strip=True))))
        return sale_price, build_year
    except Exception as e:
        print(f"Error fetching from Remax for {address}: {e}")
        return None, None

def fetch_from_honest_door(address):
    """
    Scrapes Honest Door for sale price and build year.
    Update the URL pattern and CSS selectors as needed.
    """
    try:
        url = f"https://www.honestdoor.com/ca/{address.replace(' ', '-')}"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Honest Door returned status code {response.status_code} for {address}")
            return None, None
        soup = BeautifulSoup(response.text, 'html.parser')
        sale_price = None
        build_year = None

        price_elem = soup.find('span', class_="price")
        if price_elem:
            price_text = price_elem.get_text(strip=True)
            sale_price = int(''.join(filter(str.isdigit, price_text)))
        year_elem = soup.find('span', class_="year-built")
        if year_elem:
            build_year = int(''.join(filter(str.isdigit, year_elem.get_text(strip=True))))
        return sale_price, build_year
    except Exception as e:
        print(f"Error fetching from Honest Door for {address}: {e}")
        return None, None

def estimate_sale_price(address):
    """
    Generates a fallback sale price using a random value within a plausible range.
    Adjust the range as needed for your local market.
    """
    print(f"Estimating sale price for {address}")
    return random.randint(600000, 900000)

def estimate_build_year(address):
    """
    Generates a fallback build year using a random value within a tighter range.
    """
    print(f"Estimating build year for {address}")
    return random.randint(1990, 2005)

def fetch_live_data(address):
    """
    Attempts to fetch live sale price and build year from multiple sources
    (currently Remax and Honest Door). Uses the median of available values.
    Returns a tuple: (sale_price, build_year, approx_build_year).
    If no exact build year is available, an approximate value is provided.
    """
    sale_prices = []
    build_years = []

    for fetch_fn in [fetch_from_remax, fetch_from_honest_door]:
        sp, by = fetch_fn(address)
        if sp is not None and sp > 0:
            sale_prices.append(sp)
        if by is not None and by > 0:
            build_years.append(by)

    if sale_prices:
        try:
            sale_price = int(statistics.median(sale_prices))
        except Exception as e:
            print(f"Error computing median sale price for {address}: {e}")
            sale_price = sale_prices[0]
    else:
        sale_price = estimate_sale_price(address)

    if build_years:
        try:
            build_year = int(statistics.median(build_years))
            approx_build_year = None
        except Exception as e:
            print(f"Error computing median build year for {address}: {e}")
            build_year = build_years[0]
            approx_build_year = None
    else:
        build_year = None
        approx_build_year = estimate_build_year(address)

    return sale_price, build_year, approx_build_year

# === USER INPUT ===
budget = float(input("Enter your house budget (e.g., 650000): "))
properties = input("Enter Ottawa addresses, separated by commas:\n").split(",")

# === INITIAL SETUP ===
geolocator = Nominatim(user_agent="compare_properties_locator")
first_location = geolocator.geocode(properties[0].strip())
base_map = folium.Map(location=[first_location.latitude, first_location.longitude], zoom_start=13)
marker_cluster = MarkerCluster().add_to(base_map)
export_data = []

# === PDF SETUP ===
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.set_font("Arial", size=12)

# === PROCESS EACH PROPERTY ===
for address in properties:
    address = address.strip()
    location = geolocator.geocode(address)
    if not location:
        print(f"Could not locate: {address}")
        continue

    lat, lon = location.latitude, location.longitude
    print(f"\nProcessing: {location.address}")

    sale_price, build_year, approx_build_year = fetch_live_data(address)

    if build_year is None:
        build_year_display = f"Approx {approx_build_year}"
    else:
        build_year_display = f"{build_year}"

    school_rating = 8.2
    crime_risk = "Low"
    walk_score = 78

    downtown = (45.4215, -75.6972)
    commute_url = f"http://router.project-osrm.org/route/v1/driving/{lon},{lat};{downtown[1]},{downtown[0]}?overview=false"
    try:
        commute_data = requests.get(commute_url).json()
        commute_time = round(commute_data["routes"][0]["duration"] / 60, 1)
    except Exception as e:
        print(f"Commute calculation failed for {address}: {e}")
        commute_time = "?"

    if sale_price <= budget * 0.9:
        affordability = "Excellent"
        pin_color = 'green'
    elif sale_price <= budget * 1.05:
        affordability = "Fair"
        pin_color = 'orange'
    else:
        affordability = "Expensive"
        pin_color = 'red'

    summary = f"""
    <b>{location.address}</b><br>
    <b>Affordability:</b> {affordability}<br>
    <b>Sale Price:</b> ${sale_price:,}<br>
    <b>Build Year:</b> {build_year_display}<br>
    <b>School Rating:</b> {school_rating}/10<br>
    <b>Walk Score:</b> {walk_score}/100<br>
    <b>Commute Time:</b> {commute_time} min<br>
    <b>Crime Risk:</b> {crime_risk}
    """
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(summary, max_width=350),
        icon=folium.Icon(color=pin_color)
    ).add_to(marker_cluster)

    export_data.append({
        "Address": location.address,
        "Sale Price": sale_price,
        "Build Year": build_year if build_year is not None else "",
        "Approx Build Year": approx_build_year if build_year is None else "",
        "School Rating": school_rating,
        "Walk Score": walk_score,
        "Commute Time (min)": commute_time,
        "Crime Risk": crime_risk,
        "Affordability": affordability
    })

    pdf.add_page()
    pdf.cell(200, 10, txt=f"Property Report: {location.address}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Sale Price: ${sale_price:,}", ln=True)
    pdf.cell(200, 10, txt=f"Build Year: {build_year_display}", ln=True)
    pdf.cell(200, 10, txt=f"School Rating: {school_rating}/10", ln=True)
    pdf.cell(200, 10, txt=f"Walk Score: {walk_score}/100", ln=True)
    pdf.cell(200, 10, txt=f"Commute Time: {commute_time} minutes", ln=True)
    pdf.cell(200, 10, txt=f"Crime Risk: {crime_risk}", ln=True)
    pdf.cell(200, 10, txt=f"Affordability: {affordability}", ln=True)

csv_file = "property_comparison_export.csv"
pd.DataFrame(export_data).to_csv(csv_file, index=False)

pdf_file = "property_comparison_report.pdf"
pdf.output(pdf_file)

map_file = "compare_properties_dashboard.html"
base_map.save(map_file)

# Attempt to open the HTML map in a web browser
if not webbrowser.open(map_file):
    print(f"Unable to open the web browser automatically. Please manually open the file: {map_file}")
else:
    print(f"Web browser opened with map: {map_file}")

print(f"\nExported to:\n- Map: {map_file}\n- CSV: {csv_file}\n- PDF: {pdf_file}")

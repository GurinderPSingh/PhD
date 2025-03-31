from flask import Flask, request, render_template
from flatlib.datetime import Datetime
from flatlib.geopos import GeoPos
from flatlib.chart import Chart
from flatlib import const, aspects
import datetime as dt
from geopy.geocoders import Nominatim
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Helper function to check if two objects form an applying aspect
def is_aspect_applying(obj1, obj2, orb=10):
    """Check if obj1 and obj2 are forming an applying aspect within the given orb."""
    lon1 = obj1.lon
    lon2 = obj2.lon
    diff = min((lon1 - lon2) % 360, (lon2 - lon1) % 360)  # Shortest angular distance
    major_aspects = [0, 60, 90, 120, 180]  # Conjunction, sextile, square, trine, opposition
    for aspect in major_aspects:
        if abs(diff - aspect) <= orb:
            # Check if applying (simplified: based on speed, but we'll assume applying if within orb)
            return True
    return False

# Vāstu scoring function
def vastu_score(entrance, kitchen_loc, stove_facing, sink_loc, bedroom_loc=None, toilet_loc=None, open_areas=None, slope=None, shape=None, energy=None):
    score = 0
    if entrance in ["SE", "NE"]:
        score += 30
    elif entrance == "N":
        score += 20
    else:
        score += 10
    if kitchen_loc == "NE":
        score += 30
    elif kitchen_loc in ["SE", "E"]:
        score += 20
    elif kitchen_loc == "SW":
        score += 10
    if stove_facing == "E":
        score += 20
    elif stove_facing in ["SE", "NE"]:
        score += 15
    elif stove_facing == "SW":
        score += 5
    if sink_loc in ["NE", "N"]:
        score += 20
    else:
        score += 10

    bedroom_map = {"NE": 25, "E": 20, "N": 15, "NW": 10, "SW": 5, "S": 5, "SE": 5, "W": 5}
    toilet_map = {"NW": 25, "N": 20, "W": 15, "NE": 5, "E": 5, "S": 5, "SW": 5, "SE": 5}
    open_areas_map = {"N": 20, "NE": 20, "E": 15, "NW": 10, "S": 5, "SW": 5}
    slope_map = {"NE": 20, "N": 15, "E": 15, "NW": 10, "SW": 5, "S": 5, "W": 5}
    shape_map = {"Square": 20, "Rectangle": 15, "Irregular": 5}
    energy_map = {"High": 20, "Moderate": 10, "Low": 5}

    score += bedroom_map.get(bedroom_loc, 0) if bedroom_loc else 0
    score += toilet_map.get(toilet_loc, 0) if toilet_loc else 0
    score += open_areas_map.get(open_areas, 0) if open_areas else 0
    score += slope_map.get(slope, 0) if slope else 0
    score += shape_map.get(shape, 0) if shape else 0
    score += energy_map.get(energy, 0) if energy else 0

    return min(score, 100)

# Astrology analysis with natal and transit charts
def analyze_chart_and_transits(dob, tob, birth_city, house_city, house):
    geolocator = Nominatim(user_agent="astro_app")

    # Convert dob from 'YYYY-MM-DD' to 'YYYY/MM/DD' for flatlib
    dob = dob.replace('-', '/')
    logger.debug(f"DOB: {dob}, TOB: {tob}, Birth City: {birth_city}, House City: {house_city}")

    try:
        # Natal chart
        natal_dt = Datetime(dob, tob)
        birth_loc = geolocator.geocode(birth_city)
        if not birth_loc:
            logger.error(f"Geocoding failed for birth city: {birth_city}")
            return {"error": f"Could not geocode birth city: {birth_city}"}
        natal_pos = GeoPos(birth_loc.latitude, birth_loc.longitude)
        natal_chart = Chart(natal_dt, natal_pos)

        sun_obj = natal_chart.get(const.SUN)
        logger.debug(f"Sun object: {sun_obj}, Type: {type(sun_obj)}")
        sun_sign = sun_obj.sign  # Use attribute, not method
        logger.debug(f"Sun sign: {sun_sign}")
        sun_deg = sun_obj.lon

        moon_sign = natal_chart.get(const.MOON).sign
        moon_deg = natal_chart.get(const.MOON).lon
        asc_sign = natal_chart.get(const.ASC).sign

        # Transit chart with live current date and time
        now = dt.datetime.now()
        current_date = now.strftime("%Y/%m/%d")  # e.g., "2025/03/30"
        current_time = now.strftime("%H:%M")     # e.g., "18:42"
        logger.debug(f"Transit Date: {current_date}, Transit Time: {current_time}")

        house_loc = geolocator.geocode(house_city)
        if not house_loc:
            logger.error(f"Geocoding failed for house city: {house_city}")
            return {"error": f"Could not geocode house city: {house_city}"}
        transit_pos = GeoPos(house_loc.latitude, house_loc.longitude)
        transit_dt = Datetime(current_date, current_time)
        transit_chart = Chart(transit_dt, transit_pos)

        # Real-time transits
        transits = {
            "Saturn": {"sign": transit_chart.get(const.SATURN).sign, "deg": transit_chart.get(const.SATURN).lon},
            "Jupiter": {"sign": transit_chart.get(const.JUPITER).sign, "deg": transit_chart.get(const.JUPITER).lon},
            "Mars": {"sign": transit_chart.get(const.MARS).sign, "deg": transit_chart.get(const.MARS).lon}
        }
        logger.debug(f"Transits: {transits}")

        # Impact analysis
        education = 6
        career = 6
        health = 5
        growth = 6
        finance = 5

        if sun_sign in ["Gemini", "Virgo", "Sagittarius"]:
            education += 2
        if is_aspect_applying(natal_chart.get(const.SUN), transit_chart.get(const.JUPITER)):
            education += 1
            growth += 2

        if moon_sign in ["Capricorn", "Taurus", "Virgo"]:
            career += 2
            finance += 1
        if is_aspect_applying(natal_chart.get(const.MOON), transit_chart.get(const.SATURN)):
            career -= 1
            finance -= 1

        if asc_sign in ["Aries", "Leo", "Scorpio"]:
            health += 2
        if is_aspect_applying(natal_chart.get(const.ASC), transit_chart.get(const.MARS)):
            health -= 1

        # Vāstu adjustment
        vastu = vastu_score(house["entrance"], house["kitchen_loc"], house["stove_facing"], house["sink_loc"],
                            house.get("bedroom_loc"), house.get("toilet_loc"), house.get("open_areas"),
                            house.get("slope"), house.get("shape"), house.get("energy"))
        if vastu < 50:
            finance -= 1
            health -= 1

        return {
            "sun": sun_sign, "sun_deg": sun_deg,
            "moon": moon_sign, "moon_deg": moon_deg,
            "asc": asc_sign,
            "education": min(10, education),
            "career": min(10, career),
            "health": min(10, health),
            "growth": min(10, growth),
            "finance": min(10, finance),
            "transits": transits
        }
    except Exception as e:
        logger.error(f"Error in analyze_chart_and_transits: {str(e)}")
        return {"error": str(e)}

# Recommendations (simplified, using analysis_year from form)
def get_recommendations(family_impact, sale_price, analysis_year):
    has_jupiter_boost = any(member["sun"] == "Gemini" or member["moon"] == "Gemini" for member in family_impact)
    has_saturn_risk = any(member["career"] <= 5 for member in family_impact)

    if has_jupiter_boost:
        offer_date = f"June 16-22, {analysis_year}"
    elif has_saturn_risk:
        offer_date = f"October 15-31, {analysis_year}"
    else:
        offer_date = f"July 15-31, {analysis_year}"

    offer_price = sale_price * 0.95
    avoid_period = f"March-June {analysis_year}" if has_saturn_risk else "None"
    return offer_date, offer_price, avoid_period

# Remedies
def get_remedies(house, family_impact, vastu_score):
    remedies = []
    if vastu_score < 50:
        if house["entrance"] not in ["NE", "E", "N"]:
            remedies.append("Place a Vāstu pyramid near the entrance or use a green mat.")
        if house["kitchen_loc"] not in ["NE", "SE"]:
            remedies.append("Install a water fountain in the NE kitchen corner.")
        if house.get("toilet_loc") in ["NE", "E"]:
            remedies.append("Keep toilet doors closed; use a Vāstu salt bowl.")
        if house.get("bedroom_loc") == "NE" and any(m["health"] < 5 for m in family_impact):
            remedies.append("Avoid head towards NE; use wooden bed frame.")
        if house.get("slope") not in ["NE", "N", "E"]:
            remedies.append("Level slope or place heavy objects in SW.")
        if house.get("shape") == "Irregular":
            remedies.append("Use mirrors to create virtual square/rectangle.")

    for member in family_impact:
        if member["career"] < 5:
            remedies.append(f"{member['name']}: Wear blue sapphire (consult astrologer).")
        if member["health"] < 5:
            remedies.append(f"{member['name']}: Donate red lentils on Tuesdays.")

    return remedies if remedies else ["No significant remedies needed."]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            num_houses = int(request.form["num_houses"])
            num_members = int(request.form["num_members"])
            family_income = float(request.form["family_income"])
            interest_rate = float(request.form["interest_rate"])
            special_concern = request.form["special_concern"]
            analysis_year = int(request.form.get("analysis_year", dt.datetime.now().year))

            houses = []
            family = []

            # Collect house details
            for i in range(num_houses):
                house = {
                    "address": request.form[f"address_{i}"],
                    "street": request.form[f"street_{i}"],
                    "postal_code": request.form[f"postal_code_{i}"],
                    "city": request.form[f"city_{i}"],
                    "province": request.form[f"province_{i}"],
                    "country": request.form[f"country_{i}"],
                    "sale_price": float(request.form[f"sale_price_{i}"]),
                    "link": request.form[f"link_{i}"],
                    "entrance": request.form[f"entrance_{i}"],
                    "kitchen_loc": request.form[f"kitchen_loc_{i}"],
                    "stove_facing": request.form[f"stove_facing_{i}"],
                    "sink_loc": request.form[f"sink_loc_{i}"],
                    "realtor_name": request.form.get(f"realtor_name_{i}", ""),
                    "realtor_dob": request.form.get(f"realtor_dob_{i}", ""),
                    "realtor_tob": request.form.get(f"realtor_tob_{i}", ""),
                    "realtor_city": request.form.get(f"realtor_city_{i}", ""),
                    "bedroom_loc": request.form.get(f"bedroom_loc_{i}", ""),
                    "toilet_loc": request.form.get(f"toilet_loc_{i}", ""),
                    "open_areas": request.form.get(f"open_areas_{i}", ""),
                    "slope": request.form.get(f"slope_{i}", ""),
                    "shape": request.form.get(f"shape_{i}", ""),
                    "energy": request.form.get(f"energy_{i}", "")
                }
                houses.append(house)

            # Collect family details
            for i in range(num_members):
                member = {
                    "name": request.form[f"member_name_{i}"],
                    "dob": request.form[f"member_dob_{i}"],
                    "tob": request.form[f"member_tob_{i}"],
                    "birth_city": request.form[f"member_birth_city_{i}"],
                    "current_city": request.form[f"member_current_city_{i}"]
                }
                family.append(member)

            # Analyze houses
            results = []
            for house in houses:
                vastu = vastu_score(house["entrance"], house["kitchen_loc"], house["stove_facing"], house["sink_loc"],
                                    house.get("bedroom_loc"), house.get("toilet_loc"), house.get("open_areas"),
                                    house.get("slope"), house.get("shape"), house.get("energy"))
                family_impact = []
                for member in family:
                    astro = analyze_chart_and_transits(member["dob"], member["tob"], member["birth_city"], house["city"], house)
                    if "error" in astro:
                        return render_template("error.html", error=astro["error"])
                    family_impact.append({"name": member["name"], **astro})

                offer_date, offer_price, avoid_period = get_recommendations(family_impact, house["sale_price"], analysis_year)
                remedies = get_remedies(house, family_impact, vastu)

                results.append({
                    "house": house,
                    "vastu_score": vastu,
                    "family_impact": family_impact,
                    "offer_date": offer_date,
                    "offer_price": offer_price,
                    "avoid_period": avoid_period,
                    "special_concern": special_concern,
                    "remedies": remedies,
                    "analysis_year": analysis_year
                })

            return render_template("results.html", results=results, income=family_income, rate=interest_rate)
        except Exception as e:
            logger.error(f"Error in POST handling: {str(e)}")
            return render_template("error.html", error=f"Form processing error: {str(e)}")

    # GET request: render the form
    current_year = dt.datetime.now().year
    return render_template("index.html", current_year=current_year)

if __name__ == "__main__":
    app.run(debug=True)
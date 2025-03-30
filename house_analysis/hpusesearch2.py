from flask import Flask, request, render_template
from flatlib.datetime import Datetime
from flatlib.geopos import GeoPos
from flatlib.chart import Chart
from flatlib import const, aspects
import datetime
from geopy.geocoders import Nominatim

app = Flask(__name__)

# Simplified Vāstu scoring
def vastu_score(entrance, kitchen_loc, stove_facing, sink_loc):
    score = 0
    if entrance in ["SE", "NE"]:  # Venus/Mercury - auspicious
        score += 30
    elif entrance == "N":
        score += 20
    else:  # SW, W - Rahu/Ketu risks
        score += 10
    if kitchen_loc == "NE":  # Ideal fire zone
        score += 30
    elif kitchen_loc in ["SE", "E"]:
        score += 20
    elif kitchen_loc == "SW":  # Rahu - hidden issues
        score += 10
    if stove_facing == "E":  # Fire flows east
        score += 20
    elif stove_facing in ["SE", "NE"]:
        score += 15
    elif stove_facing == "SW":  # Tension
        score += 5
    if sink_loc in ["NE", "N"]:  # Water zones
        score += 20
    else:
        score += 10
    return min(score, 100)

# Transit-based astrology analysis
def analyze_chart_and_transits(dob, tob, birth_city, house_city, house):
    # Natal chart
    dt = Datetime(dob, tob)
    geolocator = Nominatim(user_agent="astro_app")
    birth_loc = geolocator.geocode(birth_city)
    natal_pos = GeoPos(birth_loc.latitude, birth_loc.longitude)
    natal_chart = Chart(dt, natal_pos)

    sun_sign = natal_chart.get(const.SUN).sign()
    sun_deg = natal_chart.get(const.SUN).lon
    moon_sign = natal_chart.get(const.MOON).sign()
    moon_deg = natal_chart.get(const.MOON).lon
    asc_sign = natal_chart.get(const.ASC).sign()

    # Transit chart (March 29, 2025 as base, adjust for house city)
    current_date = "2025-03-29"
    current_time = "12:00"
    house_loc = geolocator.geocode(house_city)
    transit_pos = GeoPos(house_loc.latitude, house_loc.longitude)
    transit_dt = Datetime(current_date, current_time)
    transit_chart = Chart(transit_dt, transit_pos)

    # Key transits (approximate positions for 2025)
    transits = {
        "Saturn": {"sign": "Pisces", "deg": 17.0},  # March-May 2025
        "Jupiter": {"sign": "Gemini", "deg": 23.0},  # May 2025
        "Mars": {"sign": "Cancer", "deg": 10.0}     # May 2025
    }

    # Impact analysis with transits
    education = 6
    career = 6
    health = 5
    growth = 6
    finance = 5

    # Sun aspects (education, growth)
    if sun_sign in ["Gemini", "Virgo", "Sagittarius"]:
        education += 2
    if aspects.isApplying(natal_chart.get(const.SUN), transit_chart.get(const.JUPITER)):
        education += 1  # Jupiter trine boosts learning
        growth += 2

    # Moon aspects (career, finance)
    if moon_sign in ["Capricorn", "Taurus", "Virgo"]:
        career += 2
        finance += 1
    if aspects.isApplying(natal_chart.get(const.MOON), transit_chart.get(const.SATURN)):
        career -= 1  # Saturn square delays
        finance -= 1  # Financial stress

    # Ascendant aspects (health)
    if asc_sign in ["Aries", "Leo", "Scorpio"]:
        health += 2
    if aspects.isApplying(natal_chart.get(const.ASC), transit_chart.get(const.MARS)):
        health -= 1  # Mars opposition - tension

    # Adjust for house Vāstu
    vastu = vastu_score(house["entrance"], house["kitchen_loc"], house["stove_facing"], house["sink_loc"])
    if vastu < 50:
        finance -= 1  # Poor Vāstu strains budget
        health -= 1  # Tension from layout

    return {
        "sun": sun_sign, "sun_deg": sun_deg,
        "moon": moon_sign, "moon_deg": moon_deg,
        "asc": asc_sign,
        "education": min(10, education),
        "career": min(10, career),
        "health": min(10, health),
        "growth": min(10, growth),
        "finance": min(10, finance)
    }

# Offer timing based on transits
def get_offer_timing(family_impact):
    for member in family_impact:
        if member["sun"] == "Gemini" or member["moon"] == "Gemini":
            return "June 16-22, 2025"  # Jupiter in Gemini peak
        if aspects.isApplying(Datetime("2025-05-31").toJD(), Datetime("2025-03-29").toJD(), 90):  # Saturn square
            return "October 15-31, 2025"  # Post-Saturn risk
    return "July 15-31, 2025"  # Default Jupiter in Cancer

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        num_houses = int(request.form["num_houses"])
        num_members = int(request.form["num_members"])
        family_income = float(request.form["family_income"])
        interest_rate = float(request.form["interest_rate"])
        special_concern = request.form["special_concern"]

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
                "realtor_city": request.form.get(f"realtor_city_{i}", "")
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
            vastu = vastu_score(house["entrance"], house["kitchen_loc"], house["stove_facing"], house["sink_loc"])
            family_impact = []
            for member in family:
                astro = analyze_chart_and_transits(member["dob"], member["tob"], member["birth_city"], house["city"], house)
                family_impact.append({"name": member["name"], **astro})

            # Recommendations
            offer_date = get_offer_timing(family_impact)
            offer_price = house["sale_price"] * 0.95  # 5% below asking
            avoid_period = "May-October 2025" if any(aspects.isApplying(Datetime("2025-05-31").toJD(), Datetime("2025-03-29").toJD(), 90) for m in family_impact) else "None"

            results.append({
                "house": house,
                "vastu_score": vastu,
                "family_impact": family_impact,
                "offer_date": offer_date,
                "offer_price": offer_price,
                "avoid_period": avoid_period,
                "special_concern": special_concern
            })

        return render_template("results.html", results=results, income=family_income, rate=interest_rate)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
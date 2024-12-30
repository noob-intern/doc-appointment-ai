from flask import Flask, request, jsonify
from datetime import datetime, timedelta


app = Flask(__name__)


# In-memory storage for booked slots
# Key format: (doctor_id, date_string, "HH:MM") -> Boolean
# Example: bookings[(1, "2024-12-30", "17:15")] = True  # means booked
bookings = {}


def generate_timeslots(start_time, end_time, interval_minutes=15):
    """
    Generate time slots in 'HH:MM' format from start_time to end_time
    with a step of interval_minutes.
    """
    slots = []
    current_time = start_time
    while current_time < end_time:
        slots.append(current_time.strftime("%H:%M"))
        current_time += timedelta(minutes=interval_minutes)
    return slots


@app.route("/doctors/<int:doctor_id>/schedule/<string:date_str>", methods=["GET"])
def get_schedule(doctor_id, date_str):
    """
    GET: Return all 15-minute timeslots between 5 PM and 10 PM
         and show if they are booked or not.
    """
    # For simplicity, assume date_str is in YYYY-MM-DD format
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    # Set schedule range: 5 PM to 10 PM
    start_time = datetime.combine(date_obj, datetime.min.time()).replace(hour=17, minute=0)
    end_time = datetime.combine(date_obj, datetime.min.time()).replace(hour=22, minute=0)

    # Generate all timeslots
    timeslots = generate_timeslots(start_time, end_time)

    schedule_data = []
    for slot in timeslots:
        is_booked = bookings.get((doctor_id, date_str, slot), False)
        schedule_data.append({
            "time": slot,
            "status": "booked" if is_booked else "available"
        })

    return jsonify({
        "doctor_id": doctor_id,
        "date": date_str,
        "schedule": schedule_data
    })


@app.route("/doctors/<int:doctor_id>/schedule/<string:date_str>/book", methods=["POST"])
def book_timeslot(doctor_id, date_str):
    """
    POST: Book a specific 15-minute timeslot.
    Expected JSON body: {"time": "HH:MM"}
    """
    data = request.get_json()
    if not data or "time" not in data:
        return jsonify({"error": "Missing 'time' in request body"}), 400

    slot_time = data["time"]
    # Validate the slot time format (HH:MM)
    try:
        datetime.strptime(slot_time, "%H:%M")
    except ValueError:
        return jsonify({"error": "Invalid time format. Use HH:MM (24-hour format)."}), 400

    # Mark the slot as booked in our in-memory dictionary
    bookings[(doctor_id, date_str, slot_time)] = True

    return jsonify({
        "doctor_id": doctor_id,
        "date": date_str,
        "time": slot_time,
        "status": "booked"
    }), 201


if __name__ == "__main__":
    app.run(debug=True)
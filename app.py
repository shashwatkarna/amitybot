from flask import Flask, request, jsonify, render_template
import joblib
import re
import string
import json
import requests
import os
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print("🔑 Loaded API key:", GEMINI_API_KEY)

app = Flask(__name__)

def allow_iframe(response):
    response.headers["X-Frame-Options"] = "SAMEORIGIN"  
    return response

CORS(app)


#  Load directions from directions.txt
def load_directions(filepath="directions.txt"):
    directions = {}
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # print(f"📄 Loaded {len(lines)} lines from directions.txt") isko comment kiya hai
    src, dst = None, None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "from " in line.lower() and " to " in line.lower():
            try:
                parts = line.lower().split("from ")[1].split(" to ")
                src = parts[0].strip().replace("-", " ").replace("block", "").strip()
                dst = parts[1].strip().replace("-", " ").replace("block", "").strip()
                directions[(src, dst)] = []
                # print(f"➡️ Found route: {src} → {dst}") ye bhi comment kiya hai 2
            except Exception as e:
                print("⚠️ Error parsing line:", line, "Error:", e)
        elif "→" in line and src and dst:
            directions[(src, dst)].append(line)
    print("✅ Total routes loaded:", len(directions))

     #  Auto-generate reverse directions
    reverse_count = 0
    for (src, dst), steps in list(directions.items()):
        if (dst, src) not in directions:
            reverse_steps = [f"🔁 Reverse: {step}" for step in reversed(steps)]
            directions[(dst, src)] = reverse_steps
            reverse_count += 1
    print(f"🔁 Reverse routes generated: {reverse_count}")
    print("✅ Total routes loaded (with reverse):", len(directions))
    return directions

#  Load files
model = joblib.load("intent_model.pkl")
directions_data = load_directions("directions.txt")
response_dict = json.load(open("responses.json", "r", encoding="utf-8"))
location_data = json.load(open("location_info.json", "r", encoding="utf-8"))
faculty_data = json.load(open("faculty_info.json", "r", encoding="utf-8"))

#  Gemini fallback (1.5 Flash)
def gpt_fallback_response(user_message):
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": f"Answer this as if the user is asking about Amity University, Noida campus in India and in 3-4 lines:\n\n{user_message}"}]}]
    }
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        if "candidates" in result:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        print("⚠️ Gemini Error:", result)
    except Exception as e:
        print("⚠️ Gemini Exception:", e)
    return "Sorry, I'm having trouble responding right now."

#  Beautify static reply using Gemini
def gpt_rewrite_response(original_response):
    headers = {"Content-Type": "application/json"}
    prompt = f"For Amity University Noida students, rewrite this information into a concise, easy-to-read summary. Include all important details and use a natural, helpful tone. Present it with clear headings or bullet points if it makes sense:\n\n{original_response}"
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        if "candidates" in result:
            return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print("⚠️ Gemini Rewrite Exception:", e)
    return original_response

#  Detect source and destination blocks from message
def extract_blocks(message):
    patterns = [
        r'from\s+(.*?)\s+to\s+(.*)',
        r'to\s+(.*?)\s+from\s+(.*)',
        r'reach\s+(.*?)\s+from\s+(.*)',
        r'route\s+from\s+(.*?)\s+to\s+(.*)',
        r'direction\s+to\s+(.*?)\s+from\s+(.*)',
    ]
    table = str.maketrans('', '', string.punctuation)
    for pat in patterns:
        match = re.search(pat, message, re.IGNORECASE)
        if match:
            src = match.group(2 if 'from' in pat.split()[-1] else 1).lower().replace("block", "").translate(table).strip()
            dst = match.group(1 if 'from' in pat.split()[-1] else 2).lower().replace("block", "").translate(table).strip()
            return src, dst
    return None, None

#  Faculty lookup by HOD or name
def find_faculty_info(message):
    message = message.lower()

    # 1. HOD lookup by department
    if "hod" in message:
        for person in faculty_data:
            if "hod" in person.get("designation", "").lower():
                department_words = person["department"].lower().split()
                if any(word in message for word in department_words):
                    return person

    # 2. Full name match
    for person in faculty_data:
        if person["name"].lower() in message:
            return person

    # 3. Scored match based on first name + tokens
    best_match = None
    best_score = 0
    for person in faculty_data:
        score = 0
        # first name exact match = +2
        if person.get("first_name", "").lower() in message:
            score += 2
        # partial match with name tokens = +1 each
        name_tokens = re.findall(r'\w+', person["name"].lower())
        score += sum(1 for token in name_tokens if token in message)

        if score > best_score:
            best_score = score
            best_match = person

    if best_score >= 2:
        return best_match

    return None

# ✅ Location info lookup
def find_location_info(query):
    for key, val in location_data.items():
        if key.lower() in query:
            return val
    return None

# ✅ Main chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    print("📥 Received data:", data)  # 🔍 LOG HERE

    user_message = data.get("message", "")
    print("📬 Message:", user_message)

    message_lower = user_message.lower()


    #  Force Gemini fallback for broad/uncovered philosophical queries
    gemini_keywords = ["vision", "philosophy", "collaboration", "motto", "values", "international", "future", "goal"]
    if any(word in message_lower for word in gemini_keywords):
        print("🌐 Gemini fallback trigger: detected philosophical/strategic keywords.")
        reply = gpt_fallback_response(user_message)
        return jsonify({"intent": "Unknown_Intent", "response": reply})

    # 🧠 Rule-based intent overrides
    if "placement" in message_lower:
        predicted_intent = "Get_Placement_Info"
    elif "hod" in message_lower or "head of" in message_lower:
        predicted_intent = "Get_HOD_Info"
    elif "contact" in message_lower or "phone" in message_lower:
        predicted_intent = "Get_Contact_Info"
    elif any(p in message_lower for p in ["from", "to", "reach", "route", "direction"]):
        predicted_intent = "Get_Directions_To_Location"
    elif "where is" in message_lower:
        predicted_intent = "Get_Location_Info"
    elif any(person["name"].lower() in message_lower for person in faculty_data):
        predicted_intent = "Get_Faculty_Info"
    else:
        predicted_intent = model.predict([user_message])[0]

    print("📬 Message:", user_message)
    print("🎯 Predicted Intent:", predicted_intent)

    #  Directions
    if predicted_intent == "Get_Directions_To_Location":
        src, dst = extract_blocks(user_message)
        print("🧭 Extracted:", src, dst)
        if src and dst and (src, dst) in directions_data:
            steps = directions_data[(src, dst)]
            return jsonify({"intent": predicted_intent, "response": "\n".join(steps)})
        else:
            return jsonify({"intent": predicted_intent, "response": "Sorry, I couldn't find directions between those blocks."})

    #  Faculty info
    if predicted_intent in ["Get_HOD_Info", "Get_Faculty_Info"]:
        faculty = find_faculty_info(message_lower)
        if faculty:
            reply = f"{faculty['name']} is a {faculty['designation']} in {faculty['department']}. Office: {faculty['location']}. [Profile]({faculty['profile_link']})"
        else:
            reply = "Sorry, I couldn't find that faculty information."
        return jsonify({"intent": predicted_intent, "response": reply})

    # 📍 Location lookup
    # # 📍 Location lookup start hai obj obj theek karne ki
    # if predicted_intent == "Get_Location_Info":
    #     loc = find_location_info(message_lower)
    #     if loc:
    #         # Format location dictionary into user-friendly string
    #         description = loc.get("description", "")
    #         near = loc.get("near", "")
    #         contains = loc.get("contains", [])
    #         contains_text = ", ".join(contains) if contains else "No specific details available."

    #         formatted = f"{description} It is located near {near}. It includes: {contains_text}."
    #         return jsonify({"intent": predicted_intent, "response": formatted})
    #     else:
    #         return jsonify({"intent": predicted_intent, "response": gpt_fallback_response(user_message)})
    #     # ye end block abhi just daala hai object object theek karne ke liye

    # 📍 Location lookup is wale se raw string htaa rahe hai
    if predicted_intent == "Get_Location_Info":
        loc = find_location_info(message_lower)
        if loc:
            description = loc.get("description", "")
            near = loc.get("near", [])
            contains = loc.get("contains", [])

            # Format nearby and contains lists
            near_text = ", ".join(near[:-1]) + " and " + near[-1] if len(near) > 1 else (near[0] if near else "an unspecified location")
            contains_text = ", ".join(contains) if contains else "No specific details available."

            formatted = f"{description} It is located near {near_text}. It includes: {contains_text}."
            return jsonify({"intent": predicted_intent, "response": formatted})
        else:
            return jsonify({"intent": predicted_intent, "response": gpt_fallback_response(user_message)})


        

        # ye block hta dena agar nahi chala to
    #  Custom Gemini fallback trigger for "international" and abstract questions
    special_keywords = ["international", "collaboration", "foreign", "global", "exchange", "mou", "joint research", "tie-up", "partner university", "abroad", "study abroad"]
    if any(word in message_lower for word in special_keywords):
            print("🌍 Gemini fallback triggered for international-related query")
    return jsonify({"intent": "Unknown_Intent", "response": gpt_fallback_response(user_message)})
        # ye block ka end hai 


    # ✅ Static + Gemini polish or fallback
    if predicted_intent in response_dict:
        raw_response = response_dict[predicted_intent]
        trigger = ["vision", "goal", "future", "collaborate"]
        if any(k in message_lower for k in trigger):
            return jsonify({"intent": "Unknown_Intent", "response": gpt_fallback_response(user_message)})
        return jsonify({"intent": predicted_intent, "response": gpt_rewrite_response(raw_response)})

    # 🔁 Final fallback
    return jsonify({"intent": "Unknown_Intent", "response": gpt_fallback_response(user_message)})

@app.route('/map')
def show_map():
    return render_template("map.html")

@app.route('/')
def home():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)


# 3-2-1
# YE WALA CHAL RAHA HAI ye raha mera app.py isme changes nhi karni
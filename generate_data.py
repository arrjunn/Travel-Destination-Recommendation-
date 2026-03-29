"""
===================================================================
 BDA Lab Activity 6 - Data Generator
 Build a Recommendation System using Spark and MLlib
 Domain: Travel Destinations
 B.Tech CSE - 6th Semester
===================================================================
 Generates synthetic user-destination ratings data.
"""

import csv
import random
import os

random.seed(42)

# --- Travel Destination Database ---
destinations = {
    1: "Bali, Indonesia",
    2: "Santorini, Greece",
    3: "Kyoto, Japan",
    4: "Machu Picchu, Peru",
    5: "Reykjavik, Iceland",
    6: "Marrakech, Morocco",
    7: "Queenstown, New Zealand",
    8: "Amalfi Coast, Italy",
    9: "Cappadocia, Turkey",
    10: "Banff, Canada",
    11: "Maldives",
    12: "Swiss Alps, Switzerland",
    13: "Petra, Jordan",
    14: "Havana, Cuba",
    15: "Dubrovnik, Croatia",
    16: "Jaipur, India",
    17: "Patagonia, Argentina",
    18: "Bora Bora, French Polynesia",
    19: "Siem Reap, Cambodia",
    20: "Cape Town, South Africa",
    21: "Hallstatt, Austria",
    22: "Lofoten Islands, Norway",
    23: "Zhangjiajie, China",
    24: "Cusco, Peru",
    25: "Phuket, Thailand",
    26: "Zanzibar, Tanzania",
    27: "Cinque Terre, Italy",
    28: "Fiji Islands",
    29: "Chefchaouen, Morocco",
    30: "Tromsø, Norway",
    31: "Galápagos Islands, Ecuador",
    32: "Prague, Czech Republic",
    33: "Udaipur, India",
    34: "Oia, Greece",
    35: "Ha Long Bay, Vietnam",
}

# Destination categories (for generating realistic preference patterns)
CATEGORIES = {
    "beach":     [1, 11, 18, 25, 26, 28],
    "adventure": [4, 5, 7, 10, 17, 31],
    "culture":   [3, 6, 13, 14, 16, 19, 24, 32, 33],
    "scenic":    [2, 8, 9, 12, 15, 21, 22, 23, 27, 34, 35],
    "remote":    [5, 22, 26, 29, 30, 31],
}

NUM_USERS = 60
RATINGS_PER_USER = (10, 22)  # each user rates 10-22 destinations


def get_user_preference():
    """Assign each user a primary and secondary travel preference."""
    cats = list(CATEGORIES.keys())
    primary = random.choice(cats)
    secondary = random.choice([c for c in cats if c != primary])
    return primary, secondary


def generate_ratings(output_path):
    """
    Generate synthetic user-destination ratings (1.0 to 5.0).
    Users rate destinations in their preferred categories higher.
    """
    ratings = []

    for user_id in range(1, NUM_USERS + 1):
        primary, secondary = get_user_preference()
        num_ratings = random.randint(*RATINGS_PER_USER)
        rated_dests = random.sample(list(destinations.keys()), min(num_ratings, len(destinations)))

        for dest_id in rated_dests:
            # Users tend to rate preferred categories higher
            if dest_id in CATEGORIES[primary]:
                rating = round(random.uniform(3.5, 5.0), 1)
            elif dest_id in CATEGORIES[secondary]:
                rating = round(random.uniform(2.5, 4.5), 1)
            else:
                rating = round(random.uniform(1.0, 4.0), 1)
            ratings.append((user_id, dest_id, rating))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["userId", "destId", "rating"])
        writer.writerows(ratings)

    print(f"[+] Generated {len(ratings)} ratings from {NUM_USERS} users "
          f"for {len(destinations)} destinations")
    print(f"[+] Saved to: {output_path}")
    return len(ratings)


def generate_destinations_file(output_path):
    """Save destination metadata."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["destId", "name"])
        for did, name in sorted(destinations.items()):
            writer.writerow([did, name])
    print(f"[+] Destination metadata saved to: {output_path}")


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    generate_ratings(os.path.join(data_dir, "ratings.csv"))
    generate_destinations_file(os.path.join(data_dir, "destinations.csv"))
    print("\n[+] Data generation complete!")

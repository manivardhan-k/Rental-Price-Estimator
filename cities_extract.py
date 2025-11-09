import csv
import json

# Input CSV file (your dataset)
input_file = 'us_cities_states_counties.csv'
# Output JSON file
output_file = 'us-cities.json'

cities_by_state = {}

with open(input_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='|')
    
    for row in reader:
        state_abbr = row['State short'].strip()
        city_name  = row['City'].strip()
        
        if state_abbr not in cities_by_state:
            cities_by_state[state_abbr] = set()
        
        # Add the city (deduplicate automatically)
        cities_by_state[state_abbr].add(city_name)

# Convert sets to sorted lists for JSON
cities_by_state_sorted = {state: sorted(list(cities)) 
                          for state, cities in cities_by_state.items()}

# Save to JSON
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(cities_by_state_sorted, f, indent=2, ensure_ascii=False)

print(f"Saved JSON to {output_file}")

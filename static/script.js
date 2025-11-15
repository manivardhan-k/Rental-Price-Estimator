let mapInstance = null;
let marker = null;
let useAddressMode = false;

// DEFAULT US HISTORICAL PRICES
const US_HISTORICAL = {
  "2015": 1214.028173,
  "2016": 1263.230108,
  "2017": 1315.248078,
  "2018": 1369.603636,
  "2019": 1426.453892,
  "2020": 1463.329033,
  "2021": 1587.317883,
  "2022": 1782.332753,
  "2023": 1856.506433,
  "2024": 1913.500882,
  "2025": 1965.213717
};

const STATE_NAMES = {
  "AL": "Alabama",
  "AK": "Alaska",
  "AZ": "Arizona",
  "AR": "Arkansas",
  "CA": "California",
  "CO": "Colorado",
  "CT": "Connecticut",
  "DE": "Delaware",
  "FL": "Florida",
  "GA": "Georgia",
  "HI": "Hawaii",
  "ID": "Idaho",
  "IL": "Illinois",
  "IN": "Indiana",
  "IA": "Iowa",
  "KS": "Kansas",
  "KY": "Kentucky",
  "LA": "Louisiana",
  "ME": "Maine",
  "MD": "Maryland",
  "MA": "Massachusetts",
  "MI": "Michigan",
  "MN": "Minnesota",
  "MS": "Mississippi",
  "MO": "Missouri",
  "MT": "Montana",
  "NE": "Nebraska",
  "NV": "Nevada",
  "NH": "New Hampshire",
  "NJ": "New Jersey",
  "NM": "New Mexico",
  "NY": "New York",
  "NC": "North Carolina",
  "ND": "North Dakota",
  "OH": "Ohio",
  "OK": "Oklahoma",
  "OR": "Oregon",
  "PA": "Pennsylvania",
  "RI": "Rhode Island",
  "SC": "South Carolina",
  "SD": "South Dakota",
  "TN": "Tennessee",
  "TX": "Texas",
  "UT": "Utah",
  "VT": "Vermont",
  "VA": "Virginia",
  "WA": "Washington",
  "WV": "West Virginia",
  "WI": "Wisconsin",
  "WY": "Wyoming"
};


// Initialize map on page load
function initializeMap() {
  // USA bounding box
  const usBounds = [
    [24.396308, -124.848974],  // Southwest
    [49.384358, -66.885444]    // Northeast
  ];

  mapInstance = L.map('map');
  
  mapInstance.fitBounds(usBounds); // Show whole US

  // Sets the zoom level to 4
  mapInstance.setZoom(4);

  // Add OpenStreetMap tiles
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 6,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(mapInstance);

  // Do NOT add a default marker
  marker = null;
}


// Update map with new coordinates
function updateMap(lat, long, cityName = 'Property') {
  if (!mapInstance) initializeMap();

  const latNum = parseFloat(lat);
  const longNum = parseFloat(long);

  if (isNaN(latNum) || isNaN(longNum)) {
    console.error('Invalid coordinates');
    return;
  }

  // Remove old marker if exists
  if (marker) mapInstance.removeLayer(marker);

  // Add new marker
  marker = L.marker([latNum, longNum]).addTo(mapInstance)
    .bindPopup(`<strong>${cityName}</strong><br>Lat: ${latNum}<br>Long: ${longNum}`)
    .openPopup();

  // Center map on result
  mapInstance.setView([latNum, longNum], 12);
}

// Geocode address to coordinates using Nominatim API
async function geocodeAddress(street, city, state, zip) {
  try {
    // Build address string
    let addressStr = '';
    if (street) addressStr += street + ', ';
    if (city) addressStr += city + ', ';
    if (state) addressStr += state + ', ';
    if (zip) addressStr += zip;
    
    addressStr = addressStr.replace(/,\s*$/, '');
    
    const response = await fetch(`https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(addressStr)}&format=json`);
    const data = await response.json();
    
    if (data.length > 0) {
      return {
        lat: parseFloat(data[0].lat),
        long: parseFloat(data[0].lon)
      };
    } else {
      throw new Error('Address not found');
    }
  } catch (error) {
    console.error('Geocoding error:', error);
    throw error;
  }
}


// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
  // Initialize map
  initializeMap();

  // Show default US chart on page load
  createHistoricalChart(US_HISTORICAL, {}, "US");

  // ===============================
  // TOGGLE LOCATION MODE
  // ===============================
  const toggleBtn = document.getElementById('toggle-location-btn');
  const latLong = document.querySelector('.lat-long');
  const streetZip = document.querySelector('.street-zip');
  
  toggleBtn.addEventListener('click', () => {
    useAddressMode = !useAddressMode;
    if (useAddressMode) {
      latLong.style.display = 'none';
      streetZip.style.display = 'flex';
      toggleBtn.textContent = 'Use Coordinates Instead';
    } else {
      latLong.style.display = 'flex';
      streetZip.style.display = 'none';
      toggleBtn.textContent = 'Use Address Instead';
    }
  });

  // ===============================
  // AUTO-RESIZE TEXT INPUTS
  // ===============================
  const inputs = document.querySelectorAll(".input-group input[type='text']");
  inputs.forEach(input => {
    const placeholderLength = input.placeholder.length;
    input.style.width = `${placeholderLength - 1}ch`; 
  });

  // ===============================
  // STATE AND CITY DROPDOWNS
  // ===============================
  const stateSelect = document.getElementById('state');
  const citySelect = document.getElementById('city');
  
  let citiesByState = {};
  
  // Load cities from JSON file (organized by state)
  try {
    const response = await fetch('us-cities.json');
    if (!response.ok) throw new Error("City data load failed");
    citiesByState = await response.json();
  } catch (error) {
    console.error('Error loading cities:', error);
    citySelect.disabled = true;
  }
  
  // Add search/filter functionality to state select
  makeSelectSearchable(stateSelect);
  
  // Add search/filter functionality to city select
  makeSelectSearchable(citySelect);
  
  // When state changes, update city dropdown
  stateSelect.addEventListener("change", () => {
    const stateAbbr = stateSelect.value;
    const cityList = citiesByState[stateAbbr] || [];

    // Reset city dropdown
    citySelect.innerHTML = `
      <option value="" disabled selected hidden>Select City</option>
    `;

    // Populate with cities
    cityList.forEach(city => {
      const opt = document.createElement("option");
      opt.value = city;
      opt.textContent = city;
      citySelect.appendChild(opt);
    });

    // Optionally enable the city select if it was disabled
    citySelect.disabled = cityList.length === 0;
  });
  
  // ===============================
  // SEARCH BUTTON HANDLER
  // ===============================
  document.getElementById('search-btn').addEventListener('click', async () => {
    try {
      const searchBtn = document.getElementById('search-btn');
      searchBtn.textContent = 'Loading...';
      searchBtn.disabled = true;
  
      // --- Gather inputs ---
      let lat = document.getElementById('lat').value.trim();
      let long = document.getElementById('long').value.trim();
      const street = document.getElementById('street').value.trim();
      const zip = document.getElementById('zip').value.trim();
      const state = stateSelect.value.trim();
      const city = citySelect.value.trim();
      const sqft = document.getElementById('sqft').value.trim();
      const beds = document.getElementById('beds').value.trim();
      const baths = document.getElementById('baths').value.trim();
      const type = document.getElementById('type').value.trim();
  
      // --- Check mandatory fields ---
      if (!sqft || !beds || !baths || !type || !state || !city) {
        alert('Please enter all required fields: Square footage, Beds, Baths, Type, State, City');
        return;
      }
  
      if (useAddressMode) {
        if (!street && !zip) {
          alert('Please enter a street address and ZIP code');
          return;
        }
        const coords = await geocodeAddress(street, city, state, zip);
        lat = coords.lat.toString();
        long = coords.long.toString();
      }
  
      if (!lat || !long) {
        alert("Coordinates could not be determined. Please enter an address or lat/long.");
        return;
      }
  
      // Prepare data for API
      const data = {
        sqft: sqft,
        beds: beds,
        baths: baths,
        type: type,
        state: state,
        city: city,
        lat: lat,
        long: long,
        cats_allowed: document.getElementById('cats_allowed').checked ? 1 : 0,
        dogs_allowed: document.getElementById('dogs_allowed').checked ? 1 : 0,
        smoking_allowed: document.getElementById('smoking_allowed').checked ? 1 : 0,
        wheelchair_access: document.getElementById('wheelchair_access').checked ? 1 : 0,
        electric_vehicle_charge: document.getElementById('electric_vehicle_charge').checked ? 1 : 0,
        comes_furnished: document.getElementById('comes_furnished').checked ? 1 : 0,
        has_laundry: document.getElementById('has_laundry').checked ? 1 : 0,
        has_parking: document.getElementById('has_parking').checked ? 1 : 0
      };
  
      // --- Call Flask API ---
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
        mode: 'cors'
      });
  
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server error: ${response.status} - ${errorText}`);
      }
  
      const result = await response.json();
  
      if (result.success) {
        createHistoricalChart(result.historical_prices, result.predicted_prices, result.state);
        updateMap(lat, long, city);
      } else {
        alert('Error: ' + result.error);
      }
  
    } catch (error) {
      console.error('Error:', error);
      alert(error.message);
    } finally {
      const searchBtn = document.getElementById('search-btn');
      searchBtn.textContent = 'Search';
      searchBtn.disabled = false;
    }
  });
  
});

// ===============================
// CREATE ANIMATED HISTORICAL + PREDICTION CHART
// ===============================
function createHistoricalChart(historicalPrices, predictedPrices = {}, state = "US") {
  // --- Sort historical prices ---
  const sortedHist = Object.entries(historicalPrices).sort(([a], [b]) => a.localeCompare(b));
  const histYears = sortedHist.map(([year]) => year);
  const histPrices = sortedHist.map(([, price]) => price);

  // --- Sort predicted prices ---
  const sortedPred = Object.entries(predictedPrices).sort(([a], [b]) => a.localeCompare(b));
  const predYears = sortedPred.map(([year]) => year);
  const predPrices = sortedPred.map(([, price]) => price);

  // --- Dynamic axis ranges ---
  const allPrices = histPrices.concat(predPrices);
  const minPrice = Math.min(...allPrices);
  const maxPrice = Math.max(...allPrices);
  const paddingBottom = (maxPrice - minPrice) * 0.10;
  const paddingTop = (maxPrice - minPrice) * 0.10;
  const yAxisRange = [minPrice - paddingBottom, maxPrice + paddingTop];

  const allYears = histYears.concat(predYears.map(y => +y));
  const minYear = Math.min(...allYears);
  const maxYear = Math.max(...allYears);
  const xPadding = 0.75;
  const xAxisRange = [minYear - xPadding, maxYear + xPadding];

  // --- Prepare frames ---
  const frames = [];
  const maxLength = Math.max(histYears.length, predYears.length);

  const stateLabel = state === "US" ? "US Average" : `${STATE_NAMES[state] || state}\nAverage`;

  for (let i = 1; i <= maxLength; i++) {
    const currentHistYears = histYears.slice(0, i);
    const currentHistPrices = histPrices.slice(0, i);

    const currentPredYears = predYears.slice(0, i);
    const currentPredPrices = predPrices.slice(0, i);

    frames.push({
      name: i.toString(),
      data: [
        {
          x: currentHistYears,
          y: currentHistPrices,
          mode: 'lines+markers',
          line: { color: '#4f0074', width: 3 },
          marker: { size: 8, color: '#4f0074' },
          name: stateLabel
        },
        {
          x: currentPredYears,
          y: currentPredPrices,
          mode: 'lines+markers',
          line: { color: '#ff7f0e', width: 3, dash: 'dot' },
          marker: { size: 6, color: '#ff7f0e' },
          name: 'Predicted House Price'
        }
      ]
    });
  }

  // --- Initial Data (first points) ---
  const initialData = [
    {
      x: histYears,
      y: histPrices,
      mode: 'lines+markers',
      line: { color: '#4f0074', width: 3 },
      marker: { size: 8, color: '#4f0074' },
      name: stateLabel
    },
    {
      x: predYears,
      y: predPrices,
      mode: 'lines+markers',
      line: { color: '#ff7f0e', width: 3, dash: 'dot' },
      marker: { size: 6, color: '#ff7f0e' },
      name: 'Predicted House Price'
    }
  ];

  // --- Add text annotations for key points ---
  const annotations = [];

  // For US default chart (only historical data, no predicted)
  if (predYears.length === 0 && histYears.length > 0) {
    // Mark the 2025 price on historical data
    const idx2025 = histYears.indexOf('2025');
    if (idx2025 !== -1) {
      annotations.push({
        x: '2025',
        y: histPrices[idx2025],
        text: `$${Math.round(histPrices[idx2025]).toLocaleString()}`,
        showarrow: true,
        arrowhead: 2,
        arrowsize: 1,
        arrowwidth: 2,
        arrowcolor: '#4f0074',
        ax: 40,
        ay: -40,
        bgcolor: 'rgba(79, 0, 116, 0.1)',
        bordercolor: '#4f0074',
        borderwidth: 1,
        borderpad: 4,
        font: { color: '#4f0074', size: 12, family: 'Arial' }
      });
    }
  } else if (predYears.length > 0) {
    // For state data with predictions
    // Mark 2020 on historical data
    const idx2020 = histYears.indexOf('2020');
    if (idx2020 !== -1) {
      annotations.push({
        x: '2020',
        y: histPrices[idx2020],
        text: `$${Math.round(histPrices[idx2020]).toLocaleString()}`,
        showarrow: true,
        arrowhead: 2,
        arrowsize: 1,
        arrowwidth: 2,
        arrowcolor: '#4f0074',
        ax: -40,
        ay: -40,
        bgcolor: 'rgba(79, 0, 116, 0.1)',
        bordercolor: '#4f0074',
        borderwidth: 1,
        borderpad: 4,
        font: { color: '#4f0074', size: 12, family: 'Arial' }
      });
    }

    // Mark 2025 on historical data
    const idx2025Hist = histYears.indexOf('2025');
    if (idx2025Hist !== -1) {
      annotations.push({
        x: '2025',
        y: histPrices[idx2025Hist],
        text: `$${Math.round(histPrices[idx2025Hist]).toLocaleString()}`,
        showarrow: true,
        arrowhead: 2,
        arrowsize: 1,
        arrowwidth: 2,
        arrowcolor: '#4f0074',
        ax: 40,
        ay: 40,
        bgcolor: 'rgba(79, 0, 116, 0.1)',
        bordercolor: '#4f0074',
        borderwidth: 1,
        borderpad: 4,
        font: { color: '#4f0074', size: 12, family: 'Arial' }
      });
    }

    // Mark 2025 on predicted data
    const idx2025Pred = predYears.indexOf('2025');
    if (idx2025Pred !== -1) {
      annotations.push({
        x: '2025',
        y: predPrices[idx2025Pred],
        text: `$${Math.round(predPrices[idx2025Pred]).toLocaleString()}`,
        showarrow: true,
        arrowhead: 2,
        arrowsize: 1,
        arrowwidth: 2,
        arrowcolor: '#ff7f0e',
        ax: 40,
        ay: -40,
        bgcolor: 'rgba(255, 127, 14, 0.1)',
        bordercolor: '#ff7f0e',
        borderwidth: 1,
        borderpad: 4,
        font: { color: '#ff7f0e', size: 12, family: 'Arial' }
      });
    }
  }

  // --- Layout ---
  const layout = {
    title: { text: `Housing Price Trend in ${state} (${histYears[0]}–${histYears[histYears.length - 1]})`, font: { size: 18, color: '#4f0074' }},
    xaxis: { title: 'Year', tickmode: 'linear', dtick: 1, range: xAxisRange },
    yaxis: { title: 'Price ($)', tickformat: '$,.0f', range: yAxisRange, autorange: false },
    hovermode: 'closest',
    plot_bgcolor: '#f9f9f9',
    paper_bgcolor: 'white',
    margin: { t: 35, b: 60, l: 80, r: 40 },
    legend: {
      orientation: "h",
      x: 0,
      y: .92,
      xanchor: "left",
      yanchor: "bottom"
    },
    annotations: annotations
  };

  const config = { responsive: true, displayModeBar: true, displaylogo: false };

  // --- Draw chart ---
  Plotly.react('chart', initialData, layout, config).then(() => {
    // Move modebar
    const modeBar = document.querySelector('#chart .modebar-container');
    if (modeBar) {
      modeBar.style.right = '35px';
      modeBar.style.top = '361px';
    }
  });
}

// ===============================
// HELPER: MAKE SELECT SEARCHABLE
// ===============================
function makeSelectSearchable(selectElement) {
  const options = Array.from(selectElement.options);
  let searchText = '';
  let searchTimeout;
  
  selectElement.addEventListener('keydown', (e) => {
    // Clear on Escape
    if (e.key === 'Escape') {
      searchText = '';
      return;
    }
    
    // Only process letter/number keys
    if (e.key.length === 1) {
      e.preventDefault();
      
      // Add to search text
      searchText += e.key.toLowerCase();
      
      // Clear search text after 1 second of no typing
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(() => {
        searchText = '';
      }, 1000);
      
      // Find matching option
      const matchingOption = options.find(option => 
        option.textContent.toLowerCase().startsWith(searchText)
      );
      
      if (matchingOption) {
        selectElement.value = matchingOption.value;
        // Trigger change event
        selectElement.dispatchEvent(new Event('change'));
      }
    }
  });
}
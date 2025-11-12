let mapInstance = null;
let marker = null;
let useAddressMode = false;

// Initialize map on page load
function initializeMap() {
  // Default center (Flint, MI)
  const defaultLat = 42.9435;
  const defaultLong = -83.6072;
  
  mapInstance = L.map('map').setView([defaultLat, defaultLong], 12);
  
  // Add OpenStreetMap tiles
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(mapInstance);
  
  // Add default marker
  marker = L.marker([defaultLat, defaultLong]).addTo(mapInstance)
    .bindPopup('Property Location').openPopup();
}

// Update map with new coordinates
function updateMap(lat, long, cityName = 'Property') {
  if (!mapInstance) {
    initializeMap();
  }
  
  const latNum = parseFloat(lat);
  const longNum = parseFloat(long);
  
  // Validate coordinates
  if (isNaN(latNum) || isNaN(longNum)) {
    console.error('Invalid coordinates');
    return;
  }
  
  // Remove old marker
  if (marker) {
    mapInstance.removeLayer(marker);
  }
  
  // Add new marker
  marker = L.marker([latNum, longNum]).addTo(mapInstance)
    .bindPopup(`<strong>${cityName}</strong><br>Lat: ${latNum.toFixed(4)}<br>Long: ${longNum.toFixed(4)}`).openPopup();
  
  // Set map view to location with appropriate zoom
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
    // Define the default values based on your request
    const DEFAULT_VALUES = {
        sqft: '700',
        beds: '1',
        baths: '1.0',
        type: 'apartment',
        state: 'mi',
        city: 'flint',
        lat: '42.9435',
        long: '-83.6072',
        cats_allowed: 1,
        dogs_allowed: 1,
        smoking_allowed: 1,
        wheelchair_access: 0,
        electric_vehicle_charge: 0,
        comes_furnished: 0,
        has_laundry: 1,
        has_parking: 1
    };

    try {
      const searchBtn = document.getElementById('search-btn');
      searchBtn.textContent = 'Loading...';
      searchBtn.disabled = true;

      let lat = document.getElementById('lat').value || DEFAULT_VALUES.lat;
      let long = document.getElementById('long').value || DEFAULT_VALUES.long;
      let city = citySelect.value || DEFAULT_VALUES.city;

      // If using address mode, geocode the address
      if (useAddressMode) {
        const street = document.getElementById('street').value;
        const zip = document.getElementById('zip').value;
        const state = stateSelect.value || DEFAULT_VALUES.state;
        
        if (!street && !zip) {
          alert('Please enter a street address or ZIP code');
          searchBtn.textContent = 'Search';
          searchBtn.disabled = false;
          return;
        }
        
        const coords = await geocodeAddress(street, city, state, zip);
        lat = coords.lat.toString();
        long = coords.long.toString();
      }

      // --- Input Values (Read from DOM, use default if empty) ---
      const sqft = document.getElementById('sqft').value || DEFAULT_VALUES.sqft;
      const beds = document.getElementById('beds').value || DEFAULT_VALUES.beds;
      const baths = document.getElementById('baths').value || DEFAULT_VALUES.baths;
      const type = document.getElementById('type').value || DEFAULT_VALUES.type;
      const state = stateSelect.value || DEFAULT_VALUES.state;

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
      
      console.log('Sending data:', data);
      
      console.log('Sending request to Flask API...');
      
      // Call the Flask API
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
        mode: 'cors'
      });
      
      console.log('Response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server error: ${response.status} - ${errorText}`);
      }
      
      const result = await response.json();
      console.log('Response received:', result);
      
      if (result.success) {
        console.log('Prediction results:', result);
        
        // Display results
        alert(`Estimated 2020 Price: $${result.price_2020.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
Estimated 2025 Price: $${result.price_2025.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
Annual Growth Rate: ${(result.cagr * 100).toFixed(2)}%`);
        
        createHistoricalChart(result.historical_prices, result.predicted_prices, result.state);
        updateMap(lat, long, city);

      } else {
        alert('Error: ' + result.error);
      }
      
    } catch (error) {
      console.error('Detailed error:', error);
      console.error('Error stack:', error.stack);
      alert('Failed to get prediction: ' + error.message + '\n\nCheck console for details. Make sure:\n1. Flask server is running on port 5000\n2. flask-cors is installed (pip install flask-cors)');
    } finally {
      // Reset button state
      const searchBtn = document.getElementById('search-btn');
      searchBtn.textContent = 'Search';
      searchBtn.disabled = false;
    }
  });
});

// ===============================
// CREATE ANIMATED HISTORICAL + PREDICTION CHART
// ===============================
function createHistoricalChart(historicalPrices, predictedPrices, state) {
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
          name: 'State Average'
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
      x: [histYears[0]],
      y: [histPrices[0]],
      mode: 'lines+markers',
      line: { color: '#4f0074', width: 3 },
      marker: { size: 8, color: '#4f0074' },
      name: 'State Average'
    },
    {
      x: [predYears[0]],
      y: [predPrices[0]],
      mode: 'lines+markers',
      line: { color: '#ff7f0e', width: 3, dash: 'dash' },
      marker: { size: 6, color: '#ff7f0e' },
      name: 'Predicted House Price'
    }
  ];

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
    }
  };

  const config = { responsive: true, displayModeBar: true, displaylogo: false };

  // --- Draw chart and animate ---
  Plotly.react('chart', initialData, layout, config).then(() => {
    Plotly.addFrames('chart', frames);
    Plotly.animate('chart', frames.map(f => f.name), {
      frame: { duration: 300, redraw: true },
      transition: { duration: 300, easing: 'linear' },
      mode: 'immediate'
    });

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
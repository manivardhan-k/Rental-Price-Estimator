let mapInstance = null;
let marker = null;
let useAddressMode = false;
let errorTimeoutId = null;

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
  "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
  "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
  "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
  "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
  "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
  "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
  "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
  "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
  "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
  "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"
};

// ===============================
// FIELD VALIDATION SYSTEM
// ===============================
const REQUIRED_FIELDS = {
  sqft: { selector: '#sqft', label: 'Square Footage', validate: (val) => !isNaN(val) && parseFloat(val) > 0 },
  beds: { selector: '#beds', label: 'Bedrooms', validate: (val) => !isNaN(val) && parseFloat(val) > 0 },
  baths: { selector: '#baths', label: 'Bathrooms', validate: (val) => !isNaN(val) && parseFloat(val) > 0 },
  type: { selector: '#type', label: 'House Type', validate: (val) => val.trim() !== '' },
  state: { selector: '#state', label: 'State', validate: (val) => val.trim() !== '' },
  city: { selector: '#city', label: 'City', validate: (val) => val.trim() !== '' },
  lat: { selector: '#lat', label: 'Latitude', validate: (val) => !isNaN(val) && val.trim() !== '' && parseFloat(val) >= -90 && parseFloat(val) <= 90 },
  long: { selector: '#long', label: 'Longitude', validate: (val) => !isNaN(val) && val.trim() !== '' && parseFloat(val) >= -180 && parseFloat(val) <= 180 },
  street: { selector: '#street', label: 'Street', validate: (val) => val.trim() !== '' },
  zip: { selector: '#zip', label: 'ZIP', validate: (val) => val.trim() !== '' }
};

function initValidationStyles() {
  const style = document.createElement('style');
  style.textContent = `
    .field-error {
      border: 2px solid #dc2626 !important;
    }
    .field-error::placeholder {
      color: #991b1b;
    }
    .field-error option:invalid {
      color: #991b1b;
    }
  `;
  document.head.appendChild(style);
}

initValidationStyles();

function clearFieldError(fieldKey) {
  const field = document.querySelector(REQUIRED_FIELDS[fieldKey].selector);
  if (field) field.classList.remove('field-error');
}

function markFieldError(fieldKey) {
  const field = document.querySelector(REQUIRED_FIELDS[fieldKey].selector);
  if (field) field.classList.add('field-error');
}

function isFieldValid(fieldKey) {
  const field = document.querySelector(REQUIRED_FIELDS[fieldKey].selector);
  if (!field) return false;
  const value = field.value.trim();
  if (value === '' || value === null) return false;
  if (REQUIRED_FIELDS[fieldKey].validate) {
    return REQUIRED_FIELDS[fieldKey].validate(value);
  }
  return true;
}

function getActiveLocationMode() {
  const streetZipElement = document.querySelector('.street-zip');
  if (streetZipElement && streetZipElement.style.display !== 'none') {
    return 'address';
  }
  return 'coords';
}

function initFieldValidation() {
  Object.keys(REQUIRED_FIELDS).forEach(fieldKey => {
    const field = document.querySelector(REQUIRED_FIELDS[fieldKey].selector);
    if (field) {
      field.addEventListener('input', () => {
        if (isFieldValid(fieldKey)) clearFieldError(fieldKey);
      });
      field.addEventListener('change', () => {
        if (isFieldValid(fieldKey)) clearFieldError(fieldKey);
      });
      field.addEventListener('blur', () => {
        if (field.value.trim() !== '' && !isFieldValid(fieldKey)) {
          markFieldError(fieldKey);
        }
      });
    }
  });
}

function validateRequiredFieldsForMode() {
  const mode = getActiveLocationMode();
  const alwaysRequired = ['sqft', 'beds', 'baths', 'type', 'state', 'city'];
  const locationFields = mode === 'coords' ? ['lat', 'long'] : ['street', 'zip'];

  const allRequired = [...alwaysRequired, ...locationFields];

  const emptyFields = [];
  const invalidFields = [];

  // ----- Step 1: Collect empty fields -----
  allRequired.forEach(fieldKey => {
    const field = document.querySelector(REQUIRED_FIELDS[fieldKey].selector);
    if (!field) return;

    if (field.value.trim() === '') {
      markFieldError(fieldKey);
      emptyFields.push(REQUIRED_FIELDS[fieldKey].label);
    } else {
      clearFieldError(fieldKey);
    }
  });

  // If there are empty fields, show all of them
  if (emptyFields.length > 0) {
    return { hasErrors: true, message: `Please fill in: ${emptyFields.join(', ')}` };
  }

  // ----- Step 2: Collect invalid fields -----
  allRequired.forEach(fieldKey => {
    const field = document.querySelector(REQUIRED_FIELDS[fieldKey].selector);
    if (!field) return;

    const value = field.value.trim();
    if (REQUIRED_FIELDS[fieldKey].validate && !REQUIRED_FIELDS[fieldKey].validate(value)) {
      markFieldError(fieldKey);

      // Generate field-specific message
      let typeMsg = '';
      switch (fieldKey) {
        case 'sqft':
        case 'beds':
        case 'baths':
          typeMsg = 'must be a positive number.';
          break;
        case 'lat':
          typeMsg = 'must be a number between -90 and 90.';
          break;
        case 'long':
          typeMsg = 'must be a number between -180 and 180.';
          break;
        case 'zip':
          typeMsg = 'must be a valid ZIP code.';
          break;
        case 'type':
        case 'state':
        case 'city':
        case 'street':
          typeMsg = 'cannot be empty.';
          break;
        default:
          typeMsg = 'is invalid.';
      }

      invalidFields.push({ label: REQUIRED_FIELDS[fieldKey].label, message: typeMsg });
    } else {
      clearFieldError(fieldKey);
    }
  });

  // If there are invalid fields, show the first one
  if (invalidFields.length > 0) {
    const first = invalidFields[0];
    return { hasErrors: true, message: `${first.label} ${first.message}` };
  }

  // All fields valid
  return { hasErrors: false, message: '' };
}


// ===============================
// ERROR NOTIFICATION SYSTEM
// ===============================
function initErrorContainer() {
  if (!document.getElementById('error-container')) {
    const container = document.createElement('div');
    container.id = 'error-container';
    container.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 9999;
      max-width: 400px;
      font-family: Arial, sans-serif;
    `;
    document.body.appendChild(container);
  }
}

// Show error
function showError(message, duration = 5000) {
  initErrorContainer();
  const container = document.getElementById('error-container');

  // Remove existing error first
  clearError();

  const errorBox = document.createElement('div');
  errorBox.className = 'error-message';
  errorBox.innerHTML = `
    <div style="display: flex; gap: 12px; align-items: center;">
      <span style="font-size: 20px; color: #dc2626; top: 5px;">⚠</span>
      <div>
        <strong style="color: #991b1b; display: block; font-size: 0.9rem;">Error</strong>
        <p style="margin: 5px 0 0 0; color: #7f1d1d; font-size: 13px;">${message}</p>
      </div>
      <button class="error-close" style="background: none; border: none; color: #dc2626; cursor: pointer; font-size: 18px; padding: 0; line-height: 1;">×</button>
    </div>
  `;
  errorBox.style.cssText = `
    background: #fee2e2;
    border: 1px solid #fecaca;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    animation: slideIn 0.3s ease-out;
  `;

  container.appendChild(errorBox);

  // Close handler
  const closeHandler = () => {
    errorBox.style.animation = 'slideOut 0.3s ease-out';
    setTimeout(() => errorBox.remove(), 300);
  };

  // Auto remove after duration
  const timeoutId = setTimeout(() => {
    if (errorBox.parentNode) closeHandler();
  }, duration);

  // Close button
  errorBox.querySelector('.error-close').addEventListener('click', () => {
    clearTimeout(timeoutId);
    closeHandler();
  });
}

// Clear current error immediately
function clearError() {
  const container = document.getElementById('error-container');
  if (!container) return;

  const existingError = container.querySelector('.error-message');
  if (existingError) {
    existingError.remove();
  }
}

// Animation styles
const animStyle = document.createElement('style');
animStyle.textContent = `
  @keyframes slideIn {
    from { opacity: 0; transform: translateX(400px); }
    to { opacity: 1; transform: translateX(0); }
  }
  @keyframes slideOut {
    from { opacity: 1; transform: translateX(0); }
    to { opacity: 0; transform: translateX(400px); }
  }
`;
document.head.appendChild(animStyle);


// ===============================
// MAP FUNCTIONS
// ===============================
function initializeMap() {
  const usBounds = [[24.396308, -124.848974], [49.384358, -66.885444]];
  mapInstance = L.map('map');
  mapInstance.fitBounds(usBounds);
  mapInstance.setZoom(4);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 6,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(mapInstance);
  marker = null;
}

function updateMap(lat, long, cityName = 'Property') {
  if (!mapInstance) initializeMap();
  const latNum = parseFloat(lat);
  const longNum = parseFloat(long);
  if (isNaN(latNum) || isNaN(longNum)) {
    console.error('Invalid coordinates');
    return;
  }
  if (marker) mapInstance.removeLayer(marker);
  marker = L.marker([latNum, longNum]).addTo(mapInstance)
    .bindPopup(`<strong>${cityName}</strong><br>Lat: ${latNum}<br>Long: ${longNum}`)
    .openPopup();
  mapInstance.setView([latNum, longNum], 12);
}

async function geocodeAddress(street, city, state, zip) {
  try {
    let addressStr = '';
    if (street) addressStr += street + ', ';
    if (city) addressStr += city + ', ';
    if (state) addressStr += state + ', ';
    if (zip) addressStr += zip;
    addressStr = addressStr.replace(/,\s*$/, '');
    
    const response = await fetch(`https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(addressStr)}&format=json`);
    const data = await response.json();
    
    if (data.length > 0) {
      return { lat: parseFloat(data[0].lat), long: parseFloat(data[0].lon) };
    } else {
      throw new Error('Address not found');
    }
  } catch (error) {
    console.error('Geocoding error:', error);
    throw error;
  }
}

// ===============================
// MAIN INITIALIZATION
// ===============================
document.addEventListener('DOMContentLoaded', async () => {
  initializeMap();
  createHistoricalChart(US_HISTORICAL, {}, "US");
  initFieldValidation();

  // Automatically select all text when a user focuses an input
  document.querySelectorAll('input[type="text"], input[type="number"], input[type="search"]').forEach(input => {
    input.addEventListener('focus', (e) => {
      e.target.select();
    });
  });

  // TOGGLE LOCATION MODE
  const toggleBtn = document.getElementById('toggle-location-btn');
  const latLong = document.querySelector('.lat-long');
  const streetZip = document.querySelector('.street-zip');
  
  // Initialize display state
  latLong.style.display = 'flex';
  streetZip.style.display = 'none';
  
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

  // AUTO-RESIZE TEXT INPUTS
  const inputs = document.querySelectorAll(".input-group input[type='text']");
  inputs.forEach(input => {
    const placeholderLength = input.placeholder.length;
    input.style.width = `${placeholderLength - 1}ch`; 
  });

  // STATE AND CITY DROPDOWNS
  const stateSelect = document.getElementById('state');
  const citySelect = document.getElementById('city');
  let citiesByState = {};
  
  try {
    const response = await fetch('us-cities.json');
    if (!response.ok) throw new Error("City data load failed");
    citiesByState = await response.json();
  } catch (error) {
    console.error('Error loading cities:', error);
    citySelect.disabled = true;
  }
  
  makeSelectSearchable(stateSelect);
  makeSelectSearchable(citySelect);
  
  stateSelect.addEventListener("change", () => {
    const stateAbbr = stateSelect.value;
    const cityList = citiesByState[stateAbbr] || [];
    citySelect.innerHTML = `<option value="" disabled selected hidden>Select City</option>`;
    cityList.forEach(city => {
      const opt = document.createElement("option");
      opt.value = city;
      opt.textContent = city;
      citySelect.appendChild(opt);
    });
    citySelect.disabled = cityList.length === 0;
  });
  
  // SEARCH BUTTON HANDLER
  document.getElementById('search-btn').addEventListener('click', async () => {
    try {

      clearError();

      const searchBtn = document.getElementById('search-btn');
      const validation = validateRequiredFieldsForMode();
      
      if (validation.hasErrors) {
        showError(validation.message);
        return;
      }
      
      searchBtn.textContent = 'Loading...';
      searchBtn.disabled = true;

      let lat = document.getElementById('lat').value.trim();
      let long = document.getElementById('long').value.trim();
      const street = document.getElementById('street').value.trim();
      const zip = document.getElementById('zip').value.trim();
      const state = document.getElementById('state').value.trim();
      const city = document.getElementById('city').value.trim();
      const sqft = document.getElementById('sqft').value.trim();
      const beds = document.getElementById('beds').value.trim();
      const baths = document.getElementById('baths').value.trim();
      const type = document.getElementById('type').value.trim();

      const useAddressModeFlag = document.querySelector('.street-zip').style.display !== 'none';

      if (useAddressModeFlag) {
        try {
          const coords = await geocodeAddress(street, city, state, zip);
          lat = coords.lat.toString();
          long = coords.long.toString();
        } catch (error) {
          showError('Address not found. Double-check the street and ZIP code, or use latitude/longitude instead.');
          return;
        }
      }

      const data = {
        sqft, beds, baths, type, state, city, lat, long,
        cats_allowed: document.getElementById('cats_allowed').checked ? 1 : 0,
        dogs_allowed: document.getElementById('dogs_allowed').checked ? 1 : 0,
        smoking_allowed: document.getElementById('smoking_allowed').checked ? 1 : 0,
        wheelchair_access: document.getElementById('wheelchair_access').checked ? 1 : 0,
        electric_vehicle_charge: document.getElementById('electric_vehicle_charge').checked ? 1 : 0,
        comes_furnished: document.getElementById('comes_furnished').checked ? 1 : 0,
        has_laundry: document.getElementById('has_laundry').checked ? 1 : 0,
        has_parking: document.getElementById('has_parking').checked ? 1 : 0
      };

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
        showError('Error: ' + result.error);
      }

    } catch (error) {
      console.error('Error:', error);
      showError(error.message);
    } finally {
      const searchBtn = document.getElementById('search-btn');
      searchBtn.textContent = 'Search';
      searchBtn.disabled = false;
    }
  });
});

// ===============================
// HELPER: MAKE SELECT SEARCHABLE
// ===============================
function makeSelectSearchable(selectElement) {
  const options = Array.from(selectElement.options);
  let searchText = '';
  let searchTimeout;
  
  selectElement.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      searchText = '';
      return;
    }
    if (e.key.length === 1) {
      e.preventDefault();
      searchText += e.key.toLowerCase();
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(() => { searchText = ''; }, 1000);
      
      const matchingOption = options.find(option => 
        option.textContent.toLowerCase().startsWith(searchText)
      );
      if (matchingOption) {
        selectElement.value = matchingOption.value;
        selectElement.dispatchEvent(new Event('change'));
      }
    }
  });
}

// ===============================
// CREATE ANIMATED CHART
// ===============================
function createHistoricalChart(historicalPrices, predictedPrices = {}, state = "US") {
  const sortedHist = Object.entries(historicalPrices).sort(([a], [b]) => a.localeCompare(b));
  const histYears = sortedHist.map(([year]) => year);
  const histPrices = sortedHist.map(([, price]) => price);

  const sortedPred = Object.entries(predictedPrices).sort(([a], [b]) => a.localeCompare(b));
  const predYears = sortedPred.map(([year]) => year);
  const predPrices = sortedPred.map(([, price]) => price);

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

  const stateLabel = state === "US" ? "US Average" : `${STATE_NAMES[state] || state} Average`;

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

  const annotations = [];

  if (predYears.length === 0 && histYears.length > 0) {
    const idx2025 = histYears.indexOf('2025');
    if (idx2025 !== -1) {
      annotations.push({
        x: '2025', y: histPrices[idx2025],
        text: `$${Math.round(histPrices[idx2025]).toLocaleString()}`,
        showarrow: true, arrowhead: 2, arrowsize: 1, arrowwidth: 2, arrowcolor: '#4f0074',
        ax: 40, ay: -40,
        bgcolor: 'rgba(79, 0, 116, 0.1)', bordercolor: '#4f0074', borderwidth: 1, borderpad: 4,
        font: { color: '#4f0074', size: 12, family: 'Arial' }
      });
    }
  } else if (predYears.length > 0) {
    const idx2020 = histYears.indexOf('2020');
    if (idx2020 !== -1) {
      annotations.push({
        x: '2020', y: histPrices[idx2020],
        text: `$${Math.round(histPrices[idx2020]).toLocaleString()}`,
        showarrow: true, arrowhead: 2, arrowsize: 1, arrowwidth: 2, arrowcolor: '#4f0074',
        ax: -40, ay: -40,
        bgcolor: 'rgba(79, 0, 116, 0.1)', bordercolor: '#4f0074', borderwidth: 1, borderpad: 4,
        font: { color: '#4f0074', size: 12, family: 'Arial' }
      });
    }

    const idx2025Hist = histYears.indexOf('2025');
    if (idx2025Hist !== -1) {
      annotations.push({
        x: '2025', y: histPrices[idx2025Hist],
        text: `$${Math.round(histPrices[idx2025Hist]).toLocaleString()}`,
        showarrow: true, arrowhead: 2, arrowsize: 1, arrowwidth: 2, arrowcolor: '#4f0074',
        ax: 40, ay: 40,
        bgcolor: 'rgba(79, 0, 116, 0.1)', bordercolor: '#4f0074', borderwidth: 1, borderpad: 4,
        font: { color: '#4f0074', size: 12, family: 'Arial' }
      });
    }

    const idx2025Pred = predYears.indexOf('2025');
    if (idx2025Pred !== -1) {
      annotations.push({
        x: '2025', y: predPrices[idx2025Pred],
        text: `$${Math.round(predPrices[idx2025Pred]).toLocaleString()}`,
        showarrow: true, arrowhead: 2, arrowsize: 1, arrowwidth: 2, arrowcolor: '#ff7f0e',
        ax: 40, ay: -40,
        bgcolor: 'rgba(255, 127, 14, 0.1)', bordercolor: '#ff7f0e', borderwidth: 1, borderpad: 4,
        font: { color: '#ff7f0e', size: 12, family: 'Arial' }
      });
    }
  }

  const layout = {
    title: { text: `Housing Price Trend in ${state} (${histYears[0]}–${histYears[histYears.length - 1]})`, font: { size: 18, color: '#4f0074' }},
    xaxis: { title: 'Year', tickmode: 'linear', dtick: 1, range: xAxisRange },
    yaxis: { title: 'Price ($)', tickformat: '$,.0f', range: yAxisRange, autorange: false },
    hovermode: 'closest',
    plot_bgcolor: '#f9f9f9',
    paper_bgcolor: 'white',
    margin: { t: 35, b: 60, l: 80, r: 40 },
    legend: { orientation: "h", x: 0, y: .92, xanchor: "left", yanchor: "bottom" },
    annotations: annotations
  };

  const config = { responsive: true, displayModeBar: true, displaylogo: false };

  Plotly.react('chart', initialData, layout, config).then(() => {
    const modeBar = document.querySelector('#chart .modebar-container');
    if (modeBar) {
      modeBar.style.right = '35px';
      modeBar.style.top = '361px';
    }
  });
}
// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
  // ===============================
  // AUTO-RESIZE TEXT INPUTS
  // ===============================
  const inputs = document.querySelectorAll(".input-group input[type='text']");
  inputs.forEach(input => {
    const placeholderLength = input.placeholder.length;
    // Add a few extra characters for padding and cursor
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
    const response = await fetch('static/us-cities.json');
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
    const sqft = document.getElementById('sqft').value;
    const beds = document.getElementById('beds').value;
    const baths = document.getElementById('baths').value;
    const type = document.getElementById('type').value;
    const state = stateSelect.value;
    const city = citySelect.value;
    const lat = document.getElementById('lat').value;
    const long = document.getElementById('long').value;
    
    // Validate required fields
    if (!sqft || !beds || !baths || !type || !state || !city || !lat || !long) {
      alert('Please fill in all required fields');
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
      // Add default values for additional features
      cats_allowed: 0,
      dogs_allowed: 0,
      smoking_allowed: 0,
      wheelchair_access: 0,
      electric_vehicle_charge: 0,
      comes_furnished: 0,
      has_laundry: 0,
      has_parking: 0
    };
    
    console.log('Sending data:', data);
    
    try {
      // Show loading state
      const searchBtn = document.getElementById('search-btn');
      searchBtn.textContent = 'Loading...';
      searchBtn.disabled = true;
      
      console.log('Sending request to Flask API...');
      
      // Call the Flask API
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
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
        
        // Display results (you can customize this)
        alert(`Estimated 2020 Price: ${result.price_2020.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
Estimated 2025 Price: ${result.price_2025.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
Annual Growth Rate: ${(result.cagr * 100).toFixed(2)}%`);
        
        // TODO: Update chart and map with results
        // updateChart(result);
        // updateMap(lat, long);
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
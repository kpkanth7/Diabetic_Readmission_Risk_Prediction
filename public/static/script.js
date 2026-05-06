document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const btn = document.getElementById('submit-btn');
    const btnText = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.loader');
    const resultPanel = document.getElementById('result-panel');
    
    // UI State: Loading
    btnText.style.display = 'none';
    loader.style.display = 'block';
    btn.disabled = true;
    
    // Gather data
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    
    // Convert numeric fields to integers
    const numericFields = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
        'num_medications', 'number_outpatient', 'number_emergency', 
        'number_inpatient', 'number_diagnoses'
    ];
    
    numericFields.forEach(field => {
        data[field] = parseInt(data[field], 10);
    });

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const result = await response.json();
        
        // Update UI with results
        document.getElementById('res-class').innerText = result.prediction === 1 ? 'High Risk' : 'Low Risk';
        document.getElementById('res-prob').innerText = (result.probability * 100).toFixed(2) + '%';
        document.getElementById('res-band').innerText = result.risk_band;
        
        // Update Progress Bar
        const progressPct = result.probability * 100;
        const progressBar = document.getElementById('progress-bar');
        progressBar.style.width = `${progressPct}%`;
        
        // Change color based on risk
        if (result.prediction === 1) {
            progressBar.style.background = 'linear-gradient(90deg, #ef4444, #f87171)';
        } else {
            progressBar.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
        }
        
        // Update Message
        const msgDiv = document.getElementById('res-message');
        if (result.prediction === 1) {
            msgDiv.className = 'alert-message alert-danger';
            msgDiv.innerText = 'This patient profile shows elevated risk of hospital readmission within 30 days.';
        } else {
            msgDiv.className = 'alert-message alert-success';
            msgDiv.innerText = 'This patient profile appears to be lower risk for 30-day hospital readmission.';
        }
        
        // Reveal result panel with animation
        resultPanel.classList.remove('hidden');
        resultPanel.scrollIntoView({ behavior: 'smooth' });
        
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while fetching the prediction. Please try again.');
    } finally {
        // Restore UI State
        btnText.style.display = 'block';
        loader.style.display = 'none';
        btn.disabled = false;
    }
});

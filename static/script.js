document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-btn');
    const resultCard = document.getElementById('result-card');
    const predictedPriceElem = document.getElementById('predicted-price');
    const resetBtn = document.getElementById('reset-btn');
    const errorToast = document.getElementById('error-toast');
    const errorMessage = document.getElementById('error-message');
    const loader = document.getElementById('loader');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // UI State
        predictBtn.disabled = true;
        loader.style.display = 'block';
        errorToast.classList.add('hidden');
        
        // Collect form data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });
            
            const result = await response.json();
            
            if (response.ok) {
                // Show result
                predictedPriceElem.innerText = result.prediction.toLocaleString();
                form.classList.add('hidden');
                resultCard.classList.remove('hidden');
            } else {
                showError(result.error || 'Server error occurred');
            }
        } catch (error) {
            showError('Unable to connect to the server. Is it running?');
            console.error('Fetch error:', error);
        } finally {
            predictBtn.disabled = false;
            loader.style.display = 'none';
        }
    });

    resetBtn.addEventListener('click', () => {
        form.classList.remove('hidden');
        resultCard.classList.add('hidden');
        form.reset();
    });

    function showError(msg) {
        errorMessage.innerText = msg;
        errorToast.classList.remove('hidden');
        
        // Hide after 5 seconds
        setTimeout(() => {
            errorToast.classList.add('hidden');
        }, 5000);
    }
});

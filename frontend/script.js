// API Configuration
const API_BASE_URL = 'http://127.0.0.1:8000';

// Check claim function
async function checkClaim() {
    const claimInput = document.getElementById('claimInput');
    const claim = claimInput.value.trim();
    
    // Validation
    if (!claim) {
        showError('Please enter a claim to fact-check.');
        return;
    }
    
    if (claim.length > 2000) {
        showError('Claim is too long. Maximum 2000 characters allowed.');
        return;
    }
    
    // Hide previous results and errors
    hideResults();
    hideError();
    
    // Show loading state
    setLoading(true);
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/verify/text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ claim: claim })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to verify claim');
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        showError(`Error: ${error.message}. Make sure the backend is running on ${API_BASE_URL}`);
    } finally {
        setLoading(false);
    }
}

// Display results
function displayResults(data) {
    const resultSection = document.getElementById('resultSection');
    
    // Display verdict
    displayVerdict(data.verdict, data.confidence);
    
    // Display evidence
    displayEvidence(data.evidence || []);
    
    // Display explanation
    document.getElementById('explanation').textContent = data.explanation || 'No explanation available.';
    
    // Display metadata
    if (data.metadata) {
        document.getElementById('metaFactual').textContent = data.metadata.factual ? 'Yes' : 'No';
        document.getElementById('metaSources').textContent = data.metadata.sources_consulted || 0;
        document.getElementById('metaTime').textContent = `${data.metadata.processing_time_ms || 0}ms`;
    }
    
    // Show result section with animation
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Display verdict with styling
function displayVerdict(verdict, confidence) {
    const verdictBadge = document.getElementById('verdictBadge');
    const confidenceProgress = document.getElementById('confidenceProgress');
    const confidenceValue = document.getElementById('confidenceValue');
    
    // Set verdict badge
    verdictBadge.textContent = verdict;
    verdictBadge.className = 'verdict-badge';
    
    switch(verdict) {
        case 'SUPPORTED':
            verdictBadge.classList.add('supported');
            break;
        case 'REFUTED':
            verdictBadge.classList.add('refuted');
            break;
        case 'NOT_ENOUGH_INFO':
            verdictBadge.classList.add('not-enough-info');
            break;
    }
    
    // Set confidence score
    const confidencePercent = Math.round(confidence * 100);
    confidenceProgress.style.width = `${confidencePercent}%`;
    confidenceValue.textContent = `${confidencePercent}%`;
}

// Display evidence sources
function displayEvidence(evidenceList) {
    const evidenceSection = document.getElementById('evidenceSection');
    const evidenceContainer = document.getElementById('evidenceList');
    
    evidenceContainer.innerHTML = '';
    
    if (!evidenceList || evidenceList.length === 0) {
        evidenceContainer.innerHTML = '<div class="no-evidence">No evidence sources were consulted for this claim.</div>';
        return;
    }
    
    evidenceList.forEach(evidence => {
        const evidenceItem = document.createElement('div');
        evidenceItem.className = 'evidence-item';
        
        const similarityPercent = Math.round((evidence.similarity || 0) * 100);
        
        evidenceItem.innerHTML = `
            <div class="evidence-header">
                <span class="evidence-source">${evidence.source || 'Unknown Source'}</span>
                <span class="evidence-similarity">Similarity: ${similarityPercent}%</span>
            </div>
            <p class="evidence-text">${evidence.text || 'No text available'}</p>
            ${evidence.url ? `<a href="${evidence.url}" target="_blank" rel="noopener noreferrer" class="evidence-link">
                View Source →
            </a>` : ''}
        `;
        
        evidenceContainer.appendChild(evidenceItem);
    });
}

// Show error message
function showError(message) {
    const errorSection = document.getElementById('errorSection');
    const errorMessage = document.getElementById('errorMessage');
    
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Hide error message
function hideError() {
    document.getElementById('errorSection').style.display = 'none';
}

// Hide results
function hideResults() {
    document.getElementById('resultSection').style.display = 'none';
}

// Set loading state
function setLoading(isLoading) {
    const button = document.getElementById('checkButton');
    const buttonText = document.getElementById('buttonText');
    const buttonLoader = document.getElementById('buttonLoader');
    const loadingMessage = document.getElementById('loadingMessage');
    
    button.disabled = isLoading;
    buttonText.style.display = isLoading ? 'none' : 'inline';
    buttonLoader.style.display = isLoading ? 'inline-block' : 'none';
    loadingMessage.style.display = isLoading ? 'flex' : 'none';
    
    // Disable textarea while loading
    document.getElementById('claimInput').disabled = isLoading;
}

// Allow Enter key to submit (Ctrl+Enter for new line)
document.getElementById('claimInput').addEventListener('keydown', function(event) {
    if (event.key === 'Enter' && !event.ctrlKey && !event.shiftKey) {
        event.preventDefault();
        checkClaim();
    }
});

// Example claims for quick testing
const exampleClaims = [
    "Paris is the capital of France",
    "The Earth is flat",
    "India is in South Asia",
    "I want to go to India"
];

// You can add a button to load example claims if needed
console.log('Fakt-Checker Demo Ready!');
console.log('Example claims you can try:', exampleClaims);

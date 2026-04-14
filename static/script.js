/**
 * AquaMetric AI - Frontend JavaScript
 * Handles file uploads, API calls, and Results visualization
 */

let analysisData = null;
let impactChart = null;
let waterChart = null;

// ============================================
// Initialize Events on Page Load
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    initializeUploadArea();
    initializeEventListeners();
});

function initializeUploadArea() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('analyzeBtn');

    // Click to select file
    uploadArea.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            analyzeBtn.disabled = false;
            uploadArea.style.borderColor = 'var(--success-color)';
            uploadArea.querySelector('p').textContent = `✅ ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        uploadArea.querySelector('.upload-hint') && (uploadArea.querySelector('.upload-hint').textContent = 'Click Analyze PDF to continue');
        }
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            analyzeBtn.disabled = false;
            uploadArea.style.borderColor = 'var(--success-color)';
            uploadArea.querySelector('p').textContent = `✅ ${files[0].name} (${(files[0].size / 1024).toFixed(1)} KB)`;
            uploadArea.querySelector('.upload-hint') && (uploadArea.querySelector('.upload-hint').textContent = 'Click Analyze PDF to continue');
        }
    });
}

function initializeEventListeners() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const newAnalysisBtn = document.getElementById('newAnalysisBtn');
    const exportBtn = document.getElementById('exportBtn');

    analyzeBtn.addEventListener('click', analyzeFile);
    newAnalysisBtn.addEventListener('click', resetAnalysis);
    exportBtn.addEventListener('click', exportResults);
}

// ============================================
// File Analysis
// ============================================

async function analyzeFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
        showError('No file selected');
        return;
    }

    // Validate file
    if (file.type !== 'application/pdf') {
        showError('Please select a valid PDF file');
        return;
    }

    if (file.size > 50 * 1024 * 1024) {
        showError('File size exceeds 50MB limit');
        return;
    }

    // Show loading state
    showLoading();

    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('file', file);

        // Send to backend
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (!response.ok) {
            showError(result.error || 'Analysis failed');
            return;
        }

        // Process results
        analysisData = result.data;
        displayResults(analysisData);
        showResults();

    } catch (error) {
        console.error('Error:', error);
        showError(`Error: ${error.message}`);
    }
}

// ============================================
// Display Results
// ============================================

function displayResults(data) {
    // Display metrics
    document.getElementById('waterUsage').textContent = formatValue(data.water_usage, true);
    document.getElementById('wueValue').textContent = formatValue(data.WUE);
    const region = data.region || 'Unknown';
    document.getElementById('regionValue').textContent = region.length > 30 ? region.substring(0, 30) + '...' : region;
    document.getElementById('regionValue').title = region;
    const risk = data.risk_level || 'Unknown';
    const riskShort = risk.includes('High') ? 'High' : risk.includes('Low') ? 'Low' : risk.includes('Medium') ? 'Medium' : risk;
    document.getElementById('riskLevel').textContent = riskShort;

    // Update risk level color
    updateRiskLevelColor(data.risk_level);

    // Display recommendations
    displayRecommendations(data.recommendations);

    // Create charts
    createCharts(data);
}

function formatValue(value, isWater = false) {
    if (value === null || value === undefined || value === 'Unknown') return '-';
    const num = parseFloat(value);
    if (isNaN(num)) return String(value).substring(0, 50);
    if (isWater) return num.toLocaleString(undefined, { maximumFractionDigits: 1 });
    return num.toFixed(2);
}

function updateRiskLevelColor(riskLevel) {
    const element = document.getElementById('riskLevel');
    element.parentElement.style.borderLeftColor = getRiskColor(riskLevel);
}

function getRiskColor(riskLevel) {
    switch (riskLevel?.toLowerCase()) {
        case 'high':
            return '#ef4444';
        case 'medium':
            return '#f59e0b';
        case 'low':
            return '#10b981';
        default:
            return '#00a8e8';
    }
}

function displayRecommendations(recommendations) {
    const list = document.getElementById('recommendationsList');
    list.innerHTML = '';

    if (!recommendations || !Array.isArray(recommendations)) {
        list.innerHTML = '<p>No recommendations available</p>';
        return;
    }

    recommendations.forEach((rec, index) => {
        const card = document.createElement('div');
        card.className = 'recommendation-card';

        const strategy = rec.strategy || 'Strategy ' + (index + 1);
        const description = rec.description || '';
        const impact = rec.impact || '10%';

        card.innerHTML = `
            <h4>${escapeHtml(strategy)}</h4>
            <p class="description">${escapeHtml(description)}</p>
            <div class="impact">Estimated Impact: ${escapeHtml(String(impact).replace('%',''))}%</div>
        `;

        list.appendChild(card);
    });
}

// ============================================
// Chart Creation
// ============================================

function createCharts(data) {
    createImpactChart(data.recommendations);
    createWaterChart(data);
}

function createImpactChart(recommendations) {
    const ctx = document.getElementById('impactChart');
    if (!ctx) return;

    // Destroy existing chart
    if (impactChart) {
        impactChart.destroy();
    }

    // Extract data
    const labels = [];
    const impacts = [];

    if (recommendations && Array.isArray(recommendations)) {
        recommendations.forEach((rec) => {
            labels.push(rec.strategy || 'Strategy');

            // Parse impact percentage
            let impact = parseInt(rec.impact) || 10;
            impacts.push(impact);
        });
    }

    if (labels.length === 0) {
        labels.push('Strategy 1', 'Strategy 2', 'Strategy 3');
        impacts.push(10, 15, 12);
    }

    impactChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Water Reduction %',
                data: impacts,
                backgroundColor: [
                    'rgba(59, 130, 246, 0.7)',
                    'rgba(16, 185, 129, 0.7)',
                    'rgba(139, 92, 246, 0.7)',
                ],
                borderColor: [
                    '#3b82f6',
                    '#10b981',
                    '#8b5cf6',
                ],
                borderWidth: 2,
                borderRadius: 6,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#8b949e',
                        font: { size: 12 },
                    },
                },
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: '#8b949e' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                },
                x: {
                    ticks: { color: '#8b949e' },
                    grid: { display: false },
                },
            },
        },
    });
}

function createWaterChart(data) {
    const ctx = document.getElementById('waterChart');
    if (!ctx) return;

    // Destroy existing chart
    if (waterChart) {
        waterChart.destroy();
    }

    // Parse water usage
    const waterUsageNum = parseFloat(data.water_usage) || 1000;
    const reduction = waterUsageNum * 0.15; // Estimated 15% reduction

    waterChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Current Usage', 'Potential Reduction'],
            datasets: [{
                data: [waterUsageNum, reduction],
                backgroundColor: [
                    'rgba(239, 68, 68, 0.75)',
                    'rgba(16, 185, 129, 0.75)',
                ],
                borderColor: [
                    '#ef4444',
                    '#10b981',
                ],
                borderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        color: '#8b949e',
                        font: { size: 12 },
                        padding: 15,
                    },
                },
            },
        },
    });
}

// ============================================
// UI State Management
// ============================================

function showLoading() {
    document.getElementById('uploadSection').classList.add('hidden');
    document.getElementById('loadingState').classList.remove('hidden');
    document.getElementById('resultsSection').classList.add('hidden');
    document.getElementById('errorSection').classList.add('hidden');
}

function showResults() {
    document.getElementById('loadingState').classList.add('hidden');
    document.getElementById('resultsSection').classList.remove('hidden');
    document.getElementById('errorSection').classList.add('hidden');
}

function showError(message) {
    document.getElementById('loadingState').classList.add('hidden');
    document.getElementById('resultsSection').classList.add('hidden');
    document.getElementById('errorSection').classList.remove('hidden');
    document.getElementById('errorMessage').textContent = message;
}

function resetAnalysis() {
    document.getElementById('fileInput').value = '';

    // Restore upload section
    document.getElementById('uploadSection').classList.remove('hidden');
    document.getElementById('analyzeBtn').disabled = true;

    // Reset upload area text and styling
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.style.borderColor = '';
    uploadArea.querySelector('p').textContent = 'Drag and drop your PDF here';
    uploadArea.querySelector('.upload-hint').textContent = 'or click to select';

    document.getElementById('loadingState').classList.add('hidden');
    document.getElementById('resultsSection').classList.add('hidden');
    document.getElementById('errorSection').classList.add('hidden');

    if (impactChart) { impactChart.destroy(); impactChart = null; }
    if (waterChart) { waterChart.destroy(); waterChart = null; }

    analysisData = null;
}

// ============================================
// Export Results
// ============================================

function exportResults() {
    if (!analysisData) {
        alert('No data to export');
        return;
    }

    // Create JSON export
    const jsonStr = JSON.stringify(analysisData, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `aquametric-results-${new Date().getTime()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    // Also create CSV export
    exportCSV();
}

function exportCSV() {
    if (!analysisData) return;

    const data = analysisData;
    let csv = 'AquaMetric AI Analysis Report\n';
    csv += `Generated: ${new Date().toLocaleString()}\n\n`;

    csv += 'METRICS\n';
    csv += `Water Usage (ML/year),${data.water_usage}\n`;
    csv += `WUE (L/kWh),${data.WUE}\n`;
    csv += `Region,${data.region}\n`;
    csv += `Risk Level,${data.risk_level}\n\n`;

    csv += 'RECOMMENDATIONS\n';
    csv += 'Strategy,Description,Impact\n';

    if (data.recommendations && Array.isArray(data.recommendations)) {
        data.recommendations.forEach((rec) => {
            csv += `"${rec.strategy}","${rec.description}","${rec.impact}"\n`;
        });
    }

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `aquametric-results-${new Date().getTime()}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

// ============================================
// Utility Functions
// ============================================

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;',
    };
    return String(text).replace(/[&<>"']/g, (m) => map[m]);
}

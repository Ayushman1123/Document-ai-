// Document AI System - Frontend Application
const API_BASE = 'http://localhost:8000/api';

// State
let currentFile = null;
let currentResult = null;
let editingField = null;

// DOM Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const previewImage = document.getElementById('previewImage');
const fileName = document.getElementById('fileName');
const removeFile = document.getElementById('removeFile');
const extractBtn = document.getElementById('extractBtn');
const useAgentic = document.getElementById('useAgentic');
const resultsSection = document.getElementById('results');
const resultsGrid = document.getElementById('resultsGrid');
const aiExplanation = document.getElementById('aiExplanation');
const editModal = document.getElementById('editModal');
const toastContainer = document.getElementById('toastContainer');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadStats();
});

function setupEventListeners() {
    // Upload zone
    uploadZone.addEventListener('click', () => fileInput.click());
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    removeFile.addEventListener('click', clearFile);

    // Extract button
    extractBtn.addEventListener('click', extractFields);

    // Export buttons
    document.getElementById('exportJson')?.addEventListener('click', exportJson);
    document.getElementById('viewAnnotated')?.addEventListener('click', viewAnnotated);

    // Modal
    document.getElementById('cancelEdit')?.addEventListener('click', closeModal);
    document.getElementById('submitEdit')?.addEventListener('click', submitCorrection);
}

// Drag & Drop
function handleDragOver(e) {
    e.preventDefault();
    uploadZone.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length) handleFile(files[0]);
}

function handleFileSelect(e) {
    if (e.target.files.length) handleFile(e.target.files[0]);
}

function handleFile(file) {
    const validTypes = ['application/pdf', 'image/png', 'image/jpeg', 'image/tiff', 'image/bmp'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(pdf|png|jpg|jpeg|tiff|bmp)$/i)) {
        showToast('Invalid file type. Please upload PDF or image files.', 'error');
        return;
    }

    currentFile = file;
    fileName.textContent = file.name;

    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    } else {
        previewImage.classList.add('hidden');
    }

    uploadZone.classList.add('hidden');
    previewArea.classList.remove('hidden');
    extractBtn.disabled = false;
}

function clearFile() {
    currentFile = null;
    fileInput.value = '';
    previewImage.src = '';
    uploadZone.classList.remove('hidden');
    previewArea.classList.add('hidden');
    extractBtn.disabled = true;
}

// Extract Fields
async function extractFields() {
    if (!currentFile) return;

    extractBtn.disabled = true;
    extractBtn.innerHTML = '<span class="spinner"></span> Processing...';
    document.body.classList.add('loading');

    try {
        const formData = new FormData();
        formData.append('file', currentFile);

        const useAgenticAI = useAgentic.checked;

        // Use orchestrated endpoint for agentic AI
        const endpoint = useAgenticAI
            ? `${API_BASE}/extract/orchestrated`
            : `${API_BASE}/extract?use_agentic=false`;

        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Extraction failed');
        }

        currentResult = await response.json();

        // Handle orchestrated results differently
        if (useAgenticAI && currentResult.agentic_reasoning) {
            displayOrchestratedResults(currentResult);
        } else {
            displayResults(currentResult);
        }

        showToast('Extraction completed successfully!', 'success');
        loadStats();

    } catch (error) {
        console.error('Extraction error:', error);
        showToast(error.message || 'Failed to extract fields', 'error');
    } finally {
        extractBtn.disabled = false;
        extractBtn.innerHTML = 'Extract Fields';
        document.body.classList.remove('loading');
    }
}

// Display Orchestrated Results with Reasoning
function displayOrchestratedResults(result) {
    resultsSection.classList.remove('hidden');

    // Update meta with orchestration info
    const confidence = result.overall_confidence || 0;
    document.getElementById('overallConfidence').textContent = `${(confidence * 100).toFixed(0)}% Confidence`;
    document.getElementById('processingTime').textContent = `${result.reasoning_steps || 0} reasoning steps`;

    // Build results grid
    resultsGrid.innerHTML = '';

    const fields = [
        { key: 'dealer_name', label: 'Dealer Name', icon: '🏢', type: 'Fuzzy Match' },
        { key: 'model_name', label: 'Model Name', icon: '📋', type: 'Exact Match' },
        { key: 'horse_power', label: 'Horse Power', icon: '⚡', type: 'Numeric' },
        { key: 'asset_cost', label: 'Asset Cost', icon: '💰', type: 'Numeric' },
        { key: 'dealer_signature', label: 'Signature', icon: '✍️', type: 'Detection' },
        { key: 'dealer_stamp', label: 'Stamp', icon: '🔴', type: 'Detection' }
    ];

    fields.forEach(field => {
        const data = result.fields[field.key] || {};
        const card = createResultCard(field, data);
        resultsGrid.appendChild(card);
    });

    // Display Agentic Reasoning Chain
    displayReasoningChain(result.agentic_reasoning);

    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Display Reasoning Chain
function displayReasoningChain(reasoning) {
    if (!reasoning || !reasoning.workflow_log) {
        aiExplanation.innerHTML = '<h3>🤖 AI Reasoning</h3><p>No reasoning data available.</p>';
        return;
    }

    const agents = reasoning.agents_used || [];
    const steps = reasoning.workflow_log || [];

    let html = `
        <h3>🤖 Agentic AI Reasoning</h3>
        <div class="agents-used">
            <strong>Agents Used:</strong> 
            ${agents.map(a => `<span class="agent-badge">${a}</span>`).join(' ')}
        </div>
        <div class="reasoning-steps">
            <strong>Reasoning Chain:</strong>
    `;

    steps.forEach((step, idx) => {
        html += `
            <div class="reasoning-step">
                <div class="step-header">
                    <span class="step-num">${idx + 1}</span>
                    <span class="step-agent">${step.agent}</span>
                    <span class="step-time">${step.timestamp?.substring(11, 19) || ''}</span>
                </div>
                <div class="step-content">
                    <div class="step-thought">💭 ${step.thought}</div>
                    <div class="step-action">⚡ ${step.action}</div>
                    <div class="step-result">✅ ${step.result}</div>
                </div>
            </div>
        `;
    });

    html += '</div>';

    aiExplanation.innerHTML = html;
    aiExplanation.classList.remove('hidden');
}

// Display Results
function displayResults(result) {
    resultsSection.classList.remove('hidden');

    // Update meta
    const confidence = result.metadata?.overall_confidence || 0;
    document.getElementById('overallConfidence').textContent = `${(confidence * 100).toFixed(0)}% Confidence`;
    document.getElementById('processingTime').textContent = `${result.metadata?.processing_time_seconds || 0}s`;

    // Build results grid
    resultsGrid.innerHTML = '';

    const fields = [
        { key: 'dealer_name', label: 'Dealer Name', icon: '🏢', type: 'Fuzzy Match' },
        { key: 'model_name', label: 'Model Name', icon: '📋', type: 'Exact Match' },
        { key: 'horse_power', label: 'Horse Power', icon: '⚡', type: 'Numeric' },
        { key: 'asset_cost', label: 'Asset Cost', icon: '💰', type: 'Numeric' },
        { key: 'dealer_signature', label: 'Signature', icon: '✍️', type: 'Detection' },
        { key: 'dealer_stamp', label: 'Stamp', icon: '🔴', type: 'Detection' }
    ];

    fields.forEach(field => {
        const data = result.fields[field.key] || {};
        const card = createResultCard(field, data);
        resultsGrid.appendChild(card);
    });

    // AI Explanation
    if (result.explanation) {
        aiExplanation.innerHTML = `
            <h3>🤖 AI Reasoning</h3>
            <p>${result.explanation.summary || 'Extraction completed using multiple strategies.'}</p>
        `;
        aiExplanation.classList.remove('hidden');
    }

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function createResultCard(field, data) {
    const card = document.createElement('div');
    card.className = 'result-card';
    card.dataset.field = field.key;

    const isDetection = field.key.includes('signature') || field.key.includes('stamp');
    const confidence = data.confidence || 0;
    const confClass = confidence >= 0.8 ? 'high' : confidence >= 0.5 ? 'medium' : 'low';

    let valueDisplay;
    if (isDetection) {
        const present = data.present;
        valueDisplay = `
            <div class="detection-status">
                <span class="status-dot ${present ? 'present' : 'absent'}"></span>
                <span>${present ? 'Detected' : 'Not Found'}</span>
            </div>
        `;
    } else {
        const value = data.value;
        if (field.key === 'asset_cost' && value) {
            valueDisplay = `₹${Number(value).toLocaleString('en-IN')}`;
        } else if (field.key === 'horse_power' && value) {
            valueDisplay = `${value} HP`;
        } else {
            valueDisplay = value || 'Not extracted';
        }
    }

    card.innerHTML = `
        <div class="card-header">
            <div class="card-icon ${field.key.split('_')[0]}">${field.icon}</div>
            <div class="card-title">
                <h4>${field.label}</h4>
                <span class="field-type">${field.type}</span>
            </div>
        </div>
        <div class="card-value">${valueDisplay}</div>
        <div class="card-footer">
            <span class="confidence-badge confidence-${confClass}">${(confidence * 100).toFixed(0)}%</span>
            ${!isDetection ? `<button class="btn btn-secondary" onclick="openEditModal('${field.key}')">Edit</button>` : ''}
        </div>
    `;

    return card;
}

// Edit Modal
function openEditModal(fieldKey) {
    editingField = fieldKey;
    const fieldData = currentResult?.fields[fieldKey];

    document.getElementById('editFieldLabel').textContent = fieldKey.replace(/_/g, ' ').toUpperCase();
    document.getElementById('editFieldValue').value = fieldData?.value || '';

    editModal.classList.remove('hidden');
}

function closeModal() {
    editModal.classList.add('hidden');
    editingField = null;
}

async function submitCorrection() {
    if (!editingField || !currentResult) return;

    const correctValue = document.getElementById('editFieldValue').value;
    const originalValue = currentResult.fields[editingField]?.value;

    try {
        await fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                document_id: currentResult.document_id,
                field_name: editingField,
                predicted_value: String(originalValue),
                correct_value: correctValue,
                extraction_method: currentResult.fields[editingField]?.extraction_method
            })
        });

        // Update display
        currentResult.fields[editingField].value = correctValue;
        displayResults(currentResult);

        showToast('Correction submitted - AI will learn from this!', 'success');
        closeModal();

    } catch (error) {
        showToast('Failed to submit correction', 'error');
    }
}

// Export
function exportJson() {
    if (!currentResult) return;

    const blob = new Blob([JSON.stringify(currentResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `extraction_${currentResult.document_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

async function viewAnnotated() {
    if (!currentResult) return;

    try {
        const url = `${API_BASE}/results/${currentResult.document_id}/annotated`;
        window.open(url, '_blank');
    } catch (error) {
        showToast('Annotated image not available', 'error');
    }
}

// Stats
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        if (response.ok) {
            const stats = await response.json();
            document.getElementById('totalProcessed').textContent = stats.total_documents_processed || 0;
            document.getElementById('avgTime').textContent = `${(stats.average_processing_time || 0).toFixed(1)}s`;
            document.getElementById('avgConfidence').textContent = `${((stats.average_confidence || 0) * 100).toFixed(0)}%`;
        }
    } catch (error) {
        console.log('Stats not available');
    }
}

// Toast
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 4000);
}

// Document AI System - Dashboard Application
const API_BASE = 'http://localhost:8000/api';

// State
let currentFile = null;
let currentResult = null;
let editingField = null;
let isAuthenticated = false;
let currentUser = null;
let extractionHistory = [];
let currentSection = 'extract';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkAuthStatus();
    setupEventListeners();
    loadStats();
    loadHistory();
});

function setupEventListeners() {
    // Upload zone
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const removeFile = document.getElementById('removeFile');
    const extractBtn = document.getElementById('extractBtn');

    uploadZone?.addEventListener('click', () => fileInput?.click());
    uploadZone?.addEventListener('dragover', handleDragOver);
    uploadZone?.addEventListener('dragleave', handleDragLeave);
    uploadZone?.addEventListener('drop', handleDrop);
    fileInput?.addEventListener('change', handleFileSelect);
    removeFile?.addEventListener('click', clearFile);
    extractBtn?.addEventListener('click', extractFields);

    // Export buttons
    document.getElementById('exportJson')?.addEventListener('click', exportJson);
    document.getElementById('viewAnnotated')?.addEventListener('click', viewAnnotated);

    // Modal
    document.getElementById('cancelEdit')?.addEventListener('click', closeModal);
    document.getElementById('submitEdit')?.addEventListener('click', submitCorrection);
}

// Section Navigation
function showSection(sectionId) {
    currentSection = sectionId;

    // Update nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.toggle('active', link.dataset.section === sectionId);
    });

    // Show/hide sections
    document.querySelectorAll('.dashboard-section').forEach(section => {
        section.classList.toggle('hidden', section.id !== sectionId);
        section.classList.toggle('active', section.id === sectionId);
    });

    // Show message if going to results without data
    if (sectionId === 'results' && !currentResult) {
        document.getElementById('emptyResults')?.classList.remove('hidden');
        document.getElementById('resultsContent')?.classList.add('hidden');
    }

    // Refresh analytics when viewing
    if (sectionId === 'analytics') {
        loadStats();
        loadHistory();
    }
}

// Authentication
function checkAuthStatus() {
    const token = localStorage.getItem('auth_token');
    const user = localStorage.getItem('user');

    if (token && user) {
        isAuthenticated = true;
        currentUser = JSON.parse(user);
        updateAuthUI();
    } else {
        // Redirect to login page
        // window.location.href = 'auth.html';
    }
}

function updateAuthUI() {
    const userNameEl = document.getElementById('userName');
    const authBtn = document.getElementById('authBtn');

    if (isAuthenticated && currentUser) {
        userNameEl.textContent = `${currentUser.first_name} ${currentUser.last_name}`;
        authBtn.textContent = 'Logout';
        authBtn.classList.add('logout');
    } else {
        userNameEl.textContent = 'Guest';
        authBtn.textContent = 'Sign In';
        authBtn.classList.remove('logout');
    }
}

function handleAuthClick() {
    if (isAuthenticated) {
        logout();
    } else {
        window.location.href = 'auth.html';
    }
}

async function logout() {
    const token = localStorage.getItem('auth_token');

    try {
        await fetch(`${API_BASE}/auth/logout`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${token}` }
        });
    } catch (e) { }

    localStorage.removeItem('auth_token');
    localStorage.removeItem('user');
    isAuthenticated = false;
    currentUser = null;
    updateAuthUI();
    showToast('Logged out successfully', 'success');
}

// Drag & Drop
function handleDragOver(e) {
    e.preventDefault();
    document.getElementById('uploadZone')?.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    document.getElementById('uploadZone')?.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    document.getElementById('uploadZone')?.classList.remove('dragover');
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
    document.getElementById('fileName').textContent = file.name;

    const previewImage = document.getElementById('previewImage');
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

    document.getElementById('uploadZone')?.classList.add('hidden');
    document.getElementById('previewArea')?.classList.remove('hidden');
    document.getElementById('extractBtn').disabled = false;

    showToast('File ready! Click "Extract Fields" to process.', 'success');
}

function clearFile() {
    currentFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('previewImage').src = '';
    document.getElementById('uploadZone')?.classList.remove('hidden');
    document.getElementById('previewArea')?.classList.add('hidden');
    document.getElementById('extractBtn').disabled = true;
}

// Extract Fields with Agentic AI Orchestration
async function extractFields() {
    if (!currentFile) return;

    const extractBtn = document.getElementById('extractBtn');
    extractBtn.disabled = true;
    extractBtn.innerHTML = '<span class="spinner"></span> Processing with AI Agents...';
    document.body.classList.add('loading');

    const startTime = Date.now();

    try {
        const formData = new FormData();
        formData.append('file', currentFile);

        const useAgenticAI = document.getElementById('useAgentic')?.checked ?? true;

        // Use orchestrated endpoint for agentic AI
        const endpoint = useAgenticAI
            ? `${API_BASE}/extract/orchestrated`
            : `${API_BASE}/extract?use_agentic=false`;

        showToast('🤖 Agentic AI processing started...', 'info');

        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Extraction failed');
        }

        currentResult = await response.json();
        currentResult.processing_time = ((Date.now() - startTime) / 1000).toFixed(2);

        // Display results
        if (useAgenticAI && currentResult.agentic_reasoning) {
            displayOrchestratedResults(currentResult);
        } else {
            displayResults(currentResult);
        }

        // Add to history
        addToHistory(currentResult);

        // Switch to results tab
        showSection('results');

        showToast('✅ Extraction completed successfully!', 'success');
        loadStats();

    } catch (error) {
        console.error('Extraction error:', error);
        showToast('❌ ' + (error.message || 'Failed to extract fields'), 'error');
    } finally {
        extractBtn.disabled = false;
        extractBtn.innerHTML = '<span class="btn-text">Extract Fields</span><span class="btn-arrow">→</span>';
        document.body.classList.remove('loading');
    }
}

// Display Orchestrated Results with Reasoning Chain
function displayOrchestratedResults(result) {
    document.getElementById('emptyResults')?.classList.add('hidden');
    document.getElementById('resultsContent')?.classList.remove('hidden');

    // Update meta
    const confidence = result.overall_confidence || 0;
    document.getElementById('overallConfidence').textContent = `${(confidence * 100).toFixed(0)}% Confidence`;
    document.getElementById('overallConfidence').className = `confidence-badge ${confidence >= 0.8 ? 'high' : confidence >= 0.5 ? 'medium' : 'low'}`;
    document.getElementById('processingTime').textContent = `${result.reasoning_steps || 0} AI reasoning steps`;

    // Build results grid
    const resultsGrid = document.getElementById('resultsGrid');
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
}

// Display Reasoning Chain
function displayReasoningChain(reasoning) {
    const aiExplanation = document.getElementById('aiExplanation');

    if (!reasoning || !reasoning.workflow_log) {
        aiExplanation.innerHTML = `
            <h3>🤖 AI Processing</h3>
            <p>Document processed successfully. No detailed reasoning available.</p>
        `;
        return;
    }

    const agents = reasoning.agents_used || [];
    const steps = reasoning.workflow_log || [];

    let html = `
        <h3>🤖 Agentic AI Reasoning Chain</h3>
        <div class="agents-used">
            <strong>AI Agents Used:</strong>
            ${agents.map(a => `<span class="agent-badge">${formatAgentName(a)}</span>`).join(' ')}
        </div>
        <div class="reasoning-steps">
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
                    <div class="step-thought">💭 <strong>Thought:</strong> ${step.thought}</div>
                    <div class="step-action">⚡ <strong>Action:</strong> ${step.action}</div>
                    <div class="step-result">✅ <strong>Result:</strong> ${step.result}</div>
                </div>
            </div>
        `;
    });

    html += '</div>';
    aiExplanation.innerHTML = html;
}

function formatAgentName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// Display Basic Results
function displayResults(result) {
    document.getElementById('emptyResults')?.classList.add('hidden');
    document.getElementById('resultsContent')?.classList.remove('hidden');

    const confidence = result.metadata?.overall_confidence || result.overall_confidence || 0;
    document.getElementById('overallConfidence').textContent = `${(confidence * 100).toFixed(0)}% Confidence`;
    document.getElementById('processingTime').textContent = `${result.processing_time || 0}s`;

    const resultsGrid = document.getElementById('resultsGrid');
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

    const aiExplanation = document.getElementById('aiExplanation');
    if (result.explanation) {
        aiExplanation.innerHTML = `
            <h3>🤖 AI Processing</h3>
            <p>${result.explanation.summary || 'Extraction completed using pattern matching and fuzzy matching strategies.'}</p>
        `;
    }
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
                <span>${present ? 'Detected ✓' : 'Not Found'}</span>
            </div>
        `;
    } else {
        const value = data.value;
        if (field.key === 'asset_cost' && value) {
            valueDisplay = `₹${Number(value).toLocaleString('en-IN')}`;
        } else if (field.key === 'horse_power' && value) {
            valueDisplay = `${value} HP`;
        } else {
            valueDisplay = value || '<span class="not-extracted">Not extracted</span>';
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
            ${!isDetection ? `<button class="btn btn-small btn-edit" onclick="openEditModal('${field.key}')">Edit</button>` : ''}
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
    document.getElementById('editModal')?.classList.remove('hidden');
}

function closeModal() {
    document.getElementById('editModal')?.classList.add('hidden');
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

        currentResult.fields[editingField].value = correctValue;
        displayResults(currentResult);

        showToast('✅ Correction submitted - AI will learn from this!', 'success');
        closeModal();

    } catch (error) {
        showToast('❌ Failed to submit correction', 'error');
    }
}

// Export
function exportJson() {
    if (!currentResult) {
        showToast('No results to export', 'error');
        return;
    }

    const blob = new Blob([JSON.stringify(currentResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `extraction_${currentResult.document_id || 'result'}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('📥 JSON exported successfully!', 'success');
}

function viewAnnotated() {
    if (!currentResult) {
        showToast('No results available', 'error');
        return;
    }

    const url = `${API_BASE}/results/${currentResult.document_id}/annotated`;
    window.open(url, '_blank');
}

// History
function addToHistory(result) {
    const historyItem = {
        id: result.document_id,
        fileName: result.file_name || currentFile?.name || 'Document',
        confidence: result.overall_confidence || result.metadata?.overall_confidence || 0,
        timestamp: new Date().toISOString(),
        processingTime: result.processing_time || 0,
        status: result.status || 'completed'
    };

    extractionHistory.unshift(historyItem);
    if (extractionHistory.length > 50) {
        extractionHistory = extractionHistory.slice(0, 50);
    }

    localStorage.setItem('extractionHistory', JSON.stringify(extractionHistory));
    updateHistoryUI();
}

function loadHistory() {
    const saved = localStorage.getItem('extractionHistory');
    if (saved) {
        extractionHistory = JSON.parse(saved);
        updateHistoryUI();
    }
}

function updateHistoryUI() {
    const historyList = document.getElementById('historyList');
    if (!historyList) return;

    if (extractionHistory.length === 0) {
        historyList.innerHTML = `
            <div class="empty-history">
                <span class="empty-icon">📄</span>
                <p>No extraction history yet. Upload a document to get started!</p>
            </div>
        `;
        return;
    }

    historyList.innerHTML = extractionHistory.slice(0, 10).map(item => {
        const confidence = (item.confidence * 100).toFixed(0);
        const confClass = item.confidence >= 0.8 ? 'high' : item.confidence >= 0.5 ? 'medium' : 'low';
        const date = new Date(item.timestamp);
        const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();

        return `
            <div class="history-item" onclick="loadHistoryItem('${item.id}')">
                <div class="history-info">
                    <span class="history-icon">📄</span>
                    <div class="history-details">
                        <h4>${item.fileName}</h4>
                        <span>${dateStr}</span>
                    </div>
                </div>
                <span class="history-confidence ${confClass}">${confidence}%</span>
            </div>
        `;
    }).join('');
}

async function loadHistoryItem(documentId) {
    try {
        const response = await fetch(`${API_BASE}/results/${documentId}`);
        if (response.ok) {
            currentResult = await response.json();
            displayResults(currentResult);
            showSection('results');
            showToast('📋 Loaded previous result', 'success');
        } else {
            showToast('Result not found', 'error');
        }
    } catch (error) {
        showToast('Failed to load result', 'error');
    }
}

// Analytics
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        if (response.ok) {
            const stats = await response.json();
            document.getElementById('totalProcessed').textContent = stats.total_documents_processed || 0;
            document.getElementById('avgTime').textContent = `${(stats.average_processing_time || 0).toFixed(1)}s`;
            document.getElementById('avgConfidence').textContent = `${((stats.average_confidence || 0) * 100).toFixed(0)}%`;

            // Update accuracy bars
            if (stats.field_accuracy) {
                updateAccuracyBars(stats.field_accuracy);
            }
        }
    } catch (error) {
        console.log('Stats not available');
    }
}

function updateAccuracyBars(fieldAccuracy) {
    Object.entries(fieldAccuracy).forEach(([field, accuracy]) => {
        const bar = document.querySelector(`.bar-fill[data-field="${field}"]`);
        const item = bar?.closest('.accuracy-bar-item');
        const valueEl = item?.querySelector('.bar-value');

        const pct = Math.round(accuracy * 100);
        if (bar) bar.style.width = `${pct}%`;
        if (valueEl) valueEl.textContent = `${pct}%`;
    });
}

// Toast
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// Global functions
window.showSection = showSection;
window.handleAuthClick = handleAuthClick;
window.openEditModal = openEditModal;
window.loadHistoryItem = loadHistoryItem;

// Script for handling file uploads and chat interactions.


// DOM elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadContent = document.getElementById('upload-content');
const uploadIcon = document.getElementById('upload-icon');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatStatus = document.getElementById('chat-status');
const apiKeyInput = document.getElementById('api-key-input');
const saveApiKeyBtn = document.getElementById('save-api-key-btn');
const helpBtn = document.getElementById('help-btn');
const helpModal = document.getElementById('help-modal');
const closeHelpBtn = document.getElementById('close-help-btn');


// state variables
let uploadedFileName = null;
let messages = [];


// notification popups section
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            ${type === 'success'
                ? '<polyline points="20 6 9 17 4 12"/>'
                : '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'}
        </svg>
        <span>${message}</span>
    `;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}


// apiKey Handling section
function saveApiKey() {
    const key = apiKeyInput.value.trim();
    if (!key) {
        showToast('Please enter a valid Gemini API key', 'error');
        return;
    }
    localStorage.setItem('gemini_api_key', key);
    showToast('Gemini API key saved successfully!');
}

if (saveApiKeyBtn) saveApiKeyBtn.addEventListener('click', saveApiKey);

window.addEventListener('DOMContentLoaded', () => {
    const savedKey = localStorage.getItem('gemini_api_key');
    if (savedKey && apiKeyInput) {
        apiKeyInput.value = savedKey;
    }
});


// api key help handling
if (helpBtn && helpModal && closeHelpBtn) {
    helpBtn.addEventListener('click', () => {
        helpModal.style.display = 'flex';
    });

    closeHelpBtn.addEventListener('click', () => {
        helpModal.style.display = 'none';
    });

    helpModal.addEventListener('click', (e) => {
        if (e.target === helpModal) helpModal.style.display = 'none';
    });
}



// File Upload Handling section
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

async function handleFile(file) {
    const apiKey = localStorage.getItem('gemini_api_key');
    if (!apiKey) {
        showToast('Please enter and save your Gemini API key before uploading.', 'error');
        return;
    }

    const validTypes = ['application/pdf', 'text/plain'];
    if (!validTypes.includes(file.type)) {
        showToast('Please upload a PDF or TXT file', 'error');
        return;
    }

    uploadContent.innerHTML = '<div class="spinner"></div><p style="font-weight: 600;">Please hold, your document is being processed...</p>';

    const formData = new FormData();
    formData.append('file', file);
    formData.append('apiKey', apiKey);

    try {
        const response = await fetch(`/upload`, {
            method: 'POST',
            body: formData
        });

        const result = await response.text();

        if (!response.ok) throw new Error(result || 'Upload failed');

        uploadedFileName = file.name;
        dropZone.classList.add('uploaded');
        uploadIcon.innerHTML = '<circle cx="12" cy="12" r="10"/><path d="M12 6v6m0 0v6m0-6h6m-6 0H6"/>';
        uploadIcon.style.color = 'var(--accent)';
        uploadContent.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color: var(--accent); margin-bottom: 1rem;">
                <polyline points="20 6 9 17 4 12"/>
            </svg>
            <p style="font-size: 1.125rem; font-weight: 600; color: var(--accent); margin-bottom: 0.5rem;">Document Uploaded</p>
            <p style="font-size: 0.875rem; color: var(--muted-foreground); display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                ${file.name}
            </p>
            <button class="btn btn-ghost" style="margin-top: 1rem;" onclick="resetUpload()">Upload another file</button>
        `;

        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatStatus.textContent = `Chatting with: ${file.name}`;
        chatMessages.innerHTML = '';
        messages = [];

        showToast('Document processed successfully!');
    } catch (error) {
        console.error('Upload error:', error);
        showToast(error.message, 'error');
        resetUpload();
    }
}

function resetUpload() {
    uploadedFileName = null;
    dropZone.classList.remove('uploaded');
    uploadIcon.innerHTML = '<path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/>';
    uploadIcon.style.color = '';
    uploadContent.innerHTML = `
        <h3>Drop your file here</h3>
        <p>or</p>
        <label for="file-input">
            <button class="btn btn-gradient" onclick="document.getElementById('file-input').click(); return false;">
                Choose File
            </button>
        </label>
        <p style="margin-top: 1rem; font-size: 0.75rem;">Supports PDF and TXT files</p>
    `;
    chatInput.disabled = true;
    sendBtn.disabled = true;
    chatStatus.textContent = 'No document uploaded';
    chatMessages.innerHTML = `
        <div class="chat-empty">
            <h3>Upload a Document First</h3><br>
            <p>Please upload a PDF or TXT file to start asking questions.</p>
        </div>
    `;
    messages = [];
    fileInput.value = '';
    showToast('Ready for a new document');
}



// Chatbot handling and replies

async function sendMessage() {
    const apiKey = localStorage.getItem('gemini_api_key');
    if (!apiKey) {
        showToast('Please enter and save your Gemini API key before chatting.', 'error');
        return;
    }

    const question = chatInput.value.trim();
    if (!question || !uploadedFileName) return;

    addMessage('user', question);
    chatInput.value = '';

    const typingId = addTypingIndicator();

    try {
        const response = await fetch(`/chat`, { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, apiKey }) 
        });

        const data = await response.json();

        removeTypingIndicator(typingId);

        if (!response.ok) throw new Error(data.error || 'Chat failed');
        addMessage('assistant', data.answer);
    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator(typingId);
        showToast(error.message, 'error');
        addMessage('assistant', 'Error: ' + error.message);
    }
}

function addMessage(role, content) {
    messages.push({ role, content });
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.innerHTML = `
        <div class="message-avatar">
            ${role === 'assistant'
                ? '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0110 0v4"/></svg>'
                : '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'}
        </div>
        <div class="message-content">${content}</div>
    `;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addTypingIndicator() {
    const id = 'typing-' + Date.now();
    const typingDiv = document.createElement('div');
    typingDiv.id = id;
    typingDiv.className = 'message assistant';
    typingDiv.innerHTML = `
        <div class="message-avatar">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                <path d="M7 11V7a5 5 0 0110 0v4"/>
            </svg>
        </div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const element = document.getElementById(id);
    if (element) element.remove();
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// End
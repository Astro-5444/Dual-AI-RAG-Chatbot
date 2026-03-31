document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const pdfSelect = document.getElementById('pdfSelect');
    const headerDocName = document.getElementById('headerDocName');
    const fileInput = document.getElementById('fileInput');
    const uploadStatus = document.getElementById('uploadStatus');
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const chatMessages = document.getElementById('chatMessages');
    const sendBtn = document.getElementById('sendBtn');
    const clearBtn = document.getElementById('clearBtn');
    const aiStatus = document.getElementById('aiStatus');
    const statusText = document.getElementById('statusText');

    // Load available PDFs
    fetchPDFs();

    // Event Listeners
    pdfSelect.addEventListener('change', () => {
        const val = pdfSelect.value;
        headerDocName.textContent = val ? val : 'Global Knowledge Base';
    });

    fileInput.addEventListener('change', uploadPDF);
    chatForm.addEventListener('submit', sendMessage);
    clearBtn.addEventListener('click', clearHistory);

    // Functions
    async function fetchPDFs() {
        try {
            const response = await fetch('/pdfs');
            const data = await response.json();
            
            // clear existing (except first)
            while (pdfSelect.options.length > 1) {
                pdfSelect.remove(1);
            }

            data.forEach(pdf => {
                const opt = document.createElement('option');
                opt.value = pdf;
                opt.textContent = pdf;
                pdfSelect.appendChild(opt);
            });
        } catch (error) {
            console.error('Failed to fetch PDFs:', error);
        }
    }

    async function uploadPDF() {
        if (!fileInput.files.length) return;
        
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);
        
        uploadStatus.textContent = 'Uploading and processing...';
        uploadStatus.className = 'status-msg text-secondary';
        fileInput.disabled = true;

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const data = JSON.parse(line);
                        if (data.type === 'progress') {
                            let cleaned = data.line.replace(/\r/g, '');
                            if (cleaned) uploadStatus.textContent = cleaned;
                        } else if (data.type === 'success') {
                            uploadStatus.textContent = `Success!`;
                            uploadStatus.className = 'status-msg status-success';
                            fetchPDFs();
                            setTimeout(() => uploadStatus.textContent = '', 3000);
                        } else if (data.type === 'error') {
                            uploadStatus.textContent = `Error: ${data.message}`;
                            uploadStatus.className = 'status-msg status-error';
                        }
                    } catch (err) {}
                }
            }
        } catch (error) {
            uploadStatus.textContent = 'Error uploading file';
            uploadStatus.className = 'status-msg status-error';
        } finally {
            fileInput.disabled = false;
            fileInput.value = '';
        }
    }

    async function clearHistory() {
        try {
            await fetch('/clear', { method: 'POST' });
            chatMessages.innerHTML = `
                <div class="message system-msg">
                    <div class="msg-bubble">
                        <i class="fa-solid fa-broom"></i> Conversation history cleared.
                    </div>
                </div>
            `;
        } catch (error) {
            console.error('Failed to clear history');
        }
    }

    async function sendMessage(e) {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (!text) return;

        // Append user message
        appendMessage(text, 'user-msg');
        
        // Prepare UI state
        chatInput.value = '';
        chatInput.disabled = true;
        sendBtn.disabled = true;
        aiStatus.style.display = 'block';
        statusText.textContent = 'Contacting AI...';

        const pdfFilter = pdfSelect.value;

        // Create AI response bubble placeholder
        const aiBubbleObj = appendMessage('', 'ai-msg');
        const bubbleContent = aiBubbleObj.bubble;
        const thinkContent = aiBubbleObj.thinkBox;
        let currentText = '';
        let currentThinking = '';

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: text, pdf_filter: pdfFilter })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.trim() === '') continue;
                    
                    try {
                        const data = JSON.parse(line);
                        
                        if (data.type === 'status') {
                            statusText.textContent = data.message;
                        } else if (data.type === 'thinking_token') {
                            currentThinking += data.content;
                            thinkContent.textContent = currentThinking;
                            scrollToBottom();
                        } else if (data.type === 'token') {
                            currentText += data.content;
                            bubbleContent.innerHTML = typeof marked !== 'undefined' ? marked.parse(currentText) : currentText;
                            scrollToBottom();
                        } else if (data.type === 'error') {
                            bubbleContent.innerHTML += `<br><span style="color:var(--danger)">[Error: ${data.message}]</span>`;
                        } else if (data.type === 'done') {
                            statusText.textContent = `Completed using ${data.chunks} chunks`;
                        }
                    } catch (err) {
                        console.error('JSON Parse error on line:', line, err);
                    }
                }
            }
        } catch (error) {
            console.error('Chat error:', error);
            bubbleContent.innerHTML = `<span style="color:var(--danger)">Connection error. Ensure your AI proxy is running.</span>`;
        } finally {
            chatInput.disabled = false;
            sendBtn.disabled = false;
            chatInput.focus();
            setTimeout(() => { aiStatus.style.display = 'none'; }, 2000);
        }
    }

    function appendMessage(text, typeClass) {
        const div = document.createElement('div');
        div.className = `message ${typeClass}`;
        
        const container = document.createElement('div');
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.width = '100%';
        
        const thinkBox = document.createElement('div');
        thinkBox.className = 'msg-thinking';
        
        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        
        if (text) {
            bubble.textContent = text;
        } else {
            // Typing effect placeholders or just empty if we'll stream into it
            bubble.innerHTML = '<span class="typing-indicator"></span>';
        }
        
        if (typeClass === 'user-msg') {
            div.appendChild(bubble);
        } else {
            container.appendChild(thinkBox);
            container.appendChild(bubble);
            div.appendChild(container);
        }
        
        chatMessages.appendChild(div);
        scrollToBottom();
        
        return { div, bubble, thinkBox };
    }

    function scrollToBottom() {
        chatMessages.scrollTo({
            top: chatMessages.scrollHeight,
            behavior: 'smooth' 
        });
    }
});

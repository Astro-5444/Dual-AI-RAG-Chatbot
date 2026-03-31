import os
import sys
import json
import subprocess
import threading
import queue
from flask import Flask, render_template, request, jsonify, Response
import webview
from werkzeug.utils import secure_filename
import httpx

# Import RAG modules
from config import PDF_DIR, API_BASE_URL
from Chatbot import ConversationHistory, extract_keywords
from Researcher import run_researcher_reasoning, collection

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.abspath(PDF_DIR)
history = ConversationHistory()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pdfs', methods=['GET'])
def get_pdfs():
    try:
        results = collection.get(include=["metadatas"])
        sources = set()
        for meta in results.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        return jsonify(list(sources))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        def generate_upload_stream():
            try:
                # Use stdbuf or similar to force unbuffered output if needed, but universal_newlines with bufsize=1 usually works
                process = subprocess.Popen(
                    [sys.executable, "ingest.py", filepath],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                for line in iter(process.stdout.readline, ''):
                    if line:
                        yield json.dumps({'type': 'progress', 'line': line.strip()}) + '\n'
                process.stdout.close()
                return_code = process.wait()
                if return_code == 0:
                    yield json.dumps({'type': 'success', 'filename': filename}) + '\n'
                else:
                    yield json.dumps({'type': 'error', 'message': f'Ingestion failed with code {return_code}'}) + '\n'
            except Exception as e:
                yield json.dumps({'type': 'error', 'message': str(e)}) + '\n'
                
        return Response(generate_upload_stream(), mimetype='application/x-ndjson')
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/clear', methods=['POST'])
def clear_history():
    history.clear()
    return jsonify({"success": True})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    pdf_filter = data.get('pdf_filter') # Empty string if no filter
    
    if not isinstance(pdf_filter, str):
        pdf_filter = ""
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
        
    def generate():
        yield json.dumps({"type": "status", "message": "Analyzing question..."}) + "\n"
        
        from Chatbot import process_follow_up
        if history.messages:
            keywords, _ = process_follow_up(question, history, [])
        else:
            keywords = extract_keywords(question, history)
            
        yield json.dumps({"type": "status", "message": f"Reasoning with AI... Keywords: {', '.join(keywords)}"}) + "\n"
        
        q = queue.Queue()
        
        def stream_callback(event_type, data):
            q.put({"type": event_type, "data": data})

        def run_thread():
            try:
                result = run_researcher_reasoning(
                    question, 
                    keywords, 
                    source_filter=pdf_filter if pdf_filter else None,
                    stream_callback=stream_callback
                )
                q.put({"type": "done", "result": result})
            except Exception as e:
                q.put({"type": "error", "error": str(e)})

        # Start reasoning in background thread
        threading.Thread(target=run_thread, daemon=True).start()
        
        ai1_result = None
        while True:
            item = q.get()
            if item["type"] == "done":
                ai1_result = item["result"]
                break
            elif item["type"] == "error":
                yield json.dumps({"type": "error", "message": f"Researcher error: {item['error']}"}) + "\n"
                return
            elif item["type"] == "status":
                yield json.dumps({"type": "status", "message": item["data"]}) + "\n"
            elif item["type"] == "thinking":
                yield json.dumps({"type": "thinking_token", "content": item["data"]}) + "\n"
            elif item["type"] == "reasoning":
                pass # We handle finalized reasoning via "done"
            
        answer = ai1_result.get("answer", "")
        confidence = ai1_result.get("confidence", "medium")
        
        # Fast exit if no answer found
        if not answer or not answer.strip() or "not found" in answer.lower():
            fallback = "I wasn't able to find that specific information in the available documents. Could you rephrase or select a different PDF?"
            yield json.dumps({"type": "token", "content": fallback}) + "\n"
            history.add_turn(question, fallback, [])
            yield json.dumps({"type": "done", "chunks": 0}) + "\n"
            return
        
        yield json.dumps({"type": "status", "message": "Generating final response..."}) + "\n"
        
        from Chatbot import CHATBOT_MODEL, CHATBOT_MAX_TOKENS, CHATBOT_TEMPERATURE, CHATBOT_RESPONSE_PROMPT
        
        recent_history = history.get_recent(8)
        messages = [
            {"role": "system", "content": CHATBOT_RESPONSE_PROMPT.format(
                ai1_answer=answer[:2000] + "..." if len(answer) > 2000 else answer,
                confidence=confidence
            )},
        ]
        
        if recent_history:
            messages.extend(recent_history)
        messages.append({"role": "user", "content": question})
        
        payload = {
            "model": CHATBOT_MODEL,
            "messages": messages,
            "max_tokens": CHATBOT_MAX_TOKENS,
            "temperature": CHATBOT_TEMPERATURE,
            "stream": True,
        }
        
        final_answer = ""
        try:
            with httpx.Client(timeout=httpx.Timeout(300, connect=10)) as client:
                with client.stream("POST", f"{API_BASE_URL}/v1/chat/completions", json=payload) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        line = line.strip()
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            try:
                                chunk_json = json.loads(line[6:])
                                choices = chunk_json.get("choices", [])
                                if not choices:
                                    continue
                                delta = choices[0].get("delta", {})
                                
                                # Show thinking if Chatbot supports it
                                reasoning = delta.get("reasoning_content", "")
                                if reasoning:
                                    # Use the same token structure to display Chatbot thoughts
                                    yield json.dumps({"type": "thinking_token", "content": reasoning}) + "\n"
                                
                                token = delta.get("content", "")
                                if token:
                                    # Hide raw <think> tags if model emits them as normal text
                                    if "<" in token or ">" in token:
                                        import re
                                        token = re.sub(r'</?think>', '', token)
                                    if token:
                                        final_answer += token
                                        yield json.dumps({"type": "token", "content": token}) + "\n"
                            except Exception as e:
                                print(f"Error parsing chunk: {e}", flush=True)
                                continue
        except Exception as e:
            yield json.dumps({"type": "error", "message": f"Chatbot connection error: {str(e)}"}) + "\n"
            return
            
        history.add_turn(question, final_answer, ai1_result.get("chunk_ids", []))
        yield json.dumps({"type": "done", "chunks": len(ai1_result.get("chunk_ids", []))}) + "\n"

    return Response(generate(), mimetype='application/x-ndjson')

def start_webview():
    webview.create_window('AI PDF RAG Chatbot', 'http://127.0.0.1:5000', width=1280, height=800)
    webview.start()

if __name__ == '__main__':
    # Start flask in a daemon thread so it shuts down when the window closes
    t = threading.Thread(target=app.run, kwargs={'port': 5000, 'use_reloader': False})
    t.daemon = True
    t.start()
    
    start_webview()

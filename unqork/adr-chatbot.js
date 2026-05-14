/* Unqork BYO (Bring Your Own) Web Component for the ADR AI Assistant. */

class AdrChatbot extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this._api = null;
        this._sessionId = null;
        this._apiBase = '';
        this._authToken = '';
    }

    initialize(api) {
        this._api = api;
        var state = api.state.currentState();

        this._apiBase = (state.apiBaseUrl || '').replace(/\/$/, '');
        this._authToken = state.authToken || '';
        this._sessionId = state.sessionId || null;

        this._render();
        this._bindEvents();

        if (api.state && api.state.subscribe) {
            var self = this;
            api.state.subscribe(function(newState) {
                if (newState.sessionId && newState.sessionId !== self._sessionId) {
                    self._sessionId = newState.sessionId;
                    self._clearMessages();
                }
            });
        }
    }

    _render() {
        this.shadowRoot.innerHTML = '<style>' +
            ':host { display: block; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }' +
            '.chat-container { display: flex; flex-direction: column; height: 500px; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; background: #fff; }' +
            '.chat-header { padding: 10px 14px; background: #cc0000; color: #fff; font-weight: 600; font-size: 14px; }' +
            '.chat-messages { flex: 1; overflow-y: auto; padding: 12px; }' +
            '.msg { margin-bottom: 12px; max-width: 85%; }' +
            '.msg.user { margin-left: auto; background: #e3f2fd; padding: 8px 12px; border-radius: 14px 14px 4px 14px; font-size: 13px; }' +
            '.msg.assistant { background: #f5f5f5; padding: 10px 14px; border-radius: 14px 14px 14px 4px; font-size: 13px; line-height: 1.5; }' +
            '.msg.assistant p { margin: 4px 0; } .msg.assistant ul, .msg.assistant ol { padding-left: 18px; margin: 4px 0; }' +
            '.fb-bar { display: flex; gap: 6px; margin-top: 6px; padding-top: 6px; border-top: 1px solid #e0e0e0; }' +
            '.fb-btn { background: none; border: 1px solid #ccc; border-radius: 4px; padding: 1px 6px; cursor: pointer; font-size: 11px; }' +
            '.fb-btn:hover { background: #eee; } .fb-btn.sel { background: #c8e6c9; border-color: #4caf50; }' +
            '.loading { display: none; padding: 6px 12px; color: #888; font-size: 12px; }' +
            '.loading.show { display: flex; align-items: center; gap: 6px; }' +
            '.spin { width: 14px; height: 14px; border: 2px solid #ddd; border-top: 2px solid #cc0000; border-radius: 50%; animation: sp .8s linear infinite; }' +
            '@keyframes sp { to { transform: rotate(360deg); } }' +
            '.input-area { display: flex; gap: 6px; padding: 10px 12px; border-top: 1px solid #e0e0e0; background: #fafafa; }' +
            '.input-area input { flex: 1; padding: 8px 12px; border: 1px solid #ddd; border-radius: 18px; font-size: 13px; outline: none; }' +
            '.input-area input:focus { border-color: #cc0000; }' +
            '.input-area button { padding: 8px 16px; background: #cc0000; color: #fff; border: none; border-radius: 18px; font-size: 13px; cursor: pointer; }' +
            '.input-area button:hover { background: #a00; } .input-area button:disabled { background: #ccc; cursor: not-allowed; }' +
            '</style>' +
            '<div class="chat-container">' +
            '<div class="chat-header">ADR AI Assistant</div>' +
            '<div class="chat-messages" id="msgs"></div>' +
            '<div class="loading" id="loading"><div class="spin"></div><span>Analyzing...</span></div>' +
            '<div class="input-area">' +
            '<input type="text" id="inp" placeholder="Ask about the ADR documents..." autocomplete="off">' +
            '<button id="btn" type="button">Send</button>' +
            '</div></div>';
    }

    _bindEvents() {
        var self = this;
        var btn = this.shadowRoot.getElementById('btn');
        var inp = this.shadowRoot.getElementById('inp');

        btn.addEventListener('click', function() {
            var msg = inp.value.trim();
            if (msg) self._send(msg);
        });

        inp.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                var msg = inp.value.trim();
                if (msg) self._send(msg);
            }
        });

        this.shadowRoot.getElementById('msgs').addEventListener('click', function(e) {
            if (e.target.classList.contains('fb-btn')) {
                var rating = e.target.dataset.rating;
                var msgId = e.target.dataset.msg;
                self._feedback(msgId, rating);
                e.target.parentNode.querySelectorAll('.fb-btn').forEach(function(b) { b.classList.remove('sel'); });
                e.target.classList.add('sel');
            }
        });
    }

    async _send(message) {
        if (!this._sessionId) {
            this._addMsg('assistant', '<p>No session active. Please initialize a session first.</p>');
            return;
        }

        this._addMsg('user', message);
        var inp = this.shadowRoot.getElementById('inp');
        var btn = this.shadowRoot.getElementById('btn');
        inp.value = '';
        btn.disabled = true;
        this.shadowRoot.getElementById('loading').classList.add('show');

        try {
            var resp = await fetch(this._apiBase + '/api/v1/sessions/' + this._sessionId + '/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + this._authToken },
                body: JSON.stringify({ message: message })
            });
            if (!resp.ok) throw new Error('API error: ' + resp.status);
            var data = await resp.json();
            this._addMsg('assistant', data.message.content_html, data.message_id);

            if (this._api && this._api.events) {
                this._api.events.emit('adr-response', { messageId: data.message_id, content: data.message.content });
            }
        } catch (err) {
            this._addMsg('assistant', '<p style="color:#c00">Error: ' + err.message + '</p>');
        } finally {
            this.shadowRoot.getElementById('loading').classList.remove('show');
            btn.disabled = false;
            inp.focus();
        }
    }

    async _feedback(messageId, rating) {
        try {
            await fetch(this._apiBase + '/api/v1/sessions/' + this._sessionId + '/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + this._authToken },
                body: JSON.stringify({ message_id: messageId, rating: rating })
            });
        } catch (err) { console.error('Feedback error:', err); }
    }

    _addMsg(role, content, messageId) {
        var msgs = this.shadowRoot.getElementById('msgs');
        var div = document.createElement('div');
        div.className = 'msg ' + role;
        if (role === 'assistant') {
            div.innerHTML = content;
            if (messageId) {
                var bar = document.createElement('div');
                bar.className = 'fb-bar';
                bar.innerHTML = '<button class="fb-btn" data-rating="positive" data-msg="' + messageId + '">&#x1F44D;</button><button class="fb-btn" data-rating="negative" data-msg="' + messageId + '">&#x1F44E;</button>';
                div.appendChild(bar);
            }
        } else {
            div.textContent = content;
        }
        msgs.appendChild(div);
        msgs.scrollTop = msgs.scrollHeight;
    }

    _clearMessages() {
        this.shadowRoot.getElementById('msgs').innerHTML = '';
    }
}

customElements.define('adr-chatbot', AdrChatbot);

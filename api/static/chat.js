(function() {
    var config = window.ADR_CONFIG;
    var messagesEl = document.getElementById('chat-messages');
    var inputEl = document.getElementById('chat-input');
    var sendBtn = document.getElementById('send-btn');
    var loadingEl = document.getElementById('chat-loading');

    function addMessage(role, content, messageId, sources) {
        var div = document.createElement('div');
        div.className = 'message ' + role;

        if (role === 'assistant') {
            div.innerHTML = content;

            if (sources && sources.length > 0) {
                var srcDiv = document.createElement('div');
                srcDiv.className = 'sources-bar';
                var toggle = document.createElement('button');
                toggle.className = 'sources-toggle';
                toggle.textContent = 'Sources (' + sources.length + ')';
                toggle.onclick = function() {
                    var list = srcDiv.querySelector('.sources-list');
                    list.style.display = list.style.display === 'none' ? 'block' : 'none';
                };
                srcDiv.appendChild(toggle);
                var list = document.createElement('div');
                list.className = 'sources-list';
                list.style.display = 'none';
                for (var i = 0; i < sources.length; i++) {
                    var s = sources[i];
                    var item = document.createElement('div');
                    item.className = 'source-item';
                    var label = s.document || 'Unknown';
                    if (s.page) label += ' (p.' + s.page + ')';
                    item.innerHTML = '<strong>' + label + '</strong>';
                    if (s.chunk_text) {
                        var snippet = document.createElement('div');
                        snippet.className = 'source-snippet';
                        snippet.textContent = s.chunk_text.substring(0, 200) + (s.chunk_text.length > 200 ? '...' : '');
                        item.appendChild(snippet);
                    }
                    list.appendChild(item);
                }
                srcDiv.appendChild(list);
                div.appendChild(srcDiv);
            }

            if (messageId) {
                var bar = document.createElement('div');
                bar.className = 'feedback-bar';
                bar.innerHTML =
                    '<button class="feedback-btn" data-rating="positive" data-msg="' + messageId + '">&#x1F44D;</button>' +
                    '<button class="feedback-btn" data-rating="negative" data-msg="' + messageId + '">&#x1F44E;</button>';
                div.appendChild(bar);
            }
        } else {
            div.textContent = content;
        }

        messagesEl.appendChild(div);
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    async function sendQuery(message) {
        addMessage('user', message);
        inputEl.value = '';
        sendBtn.disabled = true;
        loadingEl.classList.remove('hidden');

        try {
            var resp = await fetch(config.apiBase + '/api/v1/sessions/' + config.sessionId + '/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + config.authToken
                },
                body: JSON.stringify({ message: message })
            });

            if (!resp.ok) {
                throw new Error('API error: ' + resp.status);
            }

            var data = await resp.json();
            addMessage('assistant', data.message.content_html, data.message_id, data.sources);
        } catch (err) {
            addMessage('assistant', '<p style="color:#c00">Error: ' + err.message + '</p>');
        } finally {
            loadingEl.classList.add('hidden');
            sendBtn.disabled = false;
            inputEl.focus();
        }
    }

    async function sendFeedback(messageId, rating) {
        try {
            await fetch(config.apiBase + '/api/v1/sessions/' + config.sessionId + '/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + config.authToken
                },
                body: JSON.stringify({ message_id: messageId, rating: rating })
            });
        } catch (err) {
            console.error('Feedback error:', err);
        }
    }

    sendBtn.addEventListener('click', function() {
        var msg = inputEl.value.trim();
        if (msg) sendQuery(msg);
    });

    inputEl.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            var msg = inputEl.value.trim();
            if (msg) sendQuery(msg);
        }
    });

    messagesEl.addEventListener('click', function(e) {
        if (e.target.classList.contains('feedback-btn')) {
            var rating = e.target.dataset.rating;
            var msgId = e.target.dataset.msg;
            sendFeedback(msgId, rating);
            e.target.parentNode.querySelectorAll('.feedback-btn').forEach(function(b) { b.classList.remove('selected'); });
            e.target.classList.add('selected');
        }
    });

    inputEl.focus();
})();

# ADR Chatbot - Unqork BYO Component

## Registration

1. In Unqork, navigate to **Components > BYO**
2. Upload `manifest.json` and `adr-chatbot.js`
3. The component will appear as `adr-chatbot` in the component palette

## Configuration

Set these properties in the Unqork module:

| Property | Required | Description |
|----------|----------|-------------|
| `apiBaseUrl` | Yes | Base URL of the ADR AI Agent API (e.g., `https://adr-api.example.com`) |
| `authToken` | Yes | Bearer token for API authentication |
| `sessionId` | No | Set after session initialization to activate the chatbot |

## Usage

1. Drop the component onto your module canvas
2. Configure `apiBaseUrl` and `authToken`
3. Use a Plug-In component to call `POST /api/v1/sessions` and map the returned `session_id` to the component's `sessionId` property
4. The chatbot activates when `sessionId` is set

## Events

The component emits:
- `adr-response` - fired after each agent response, with `{ messageId, content }`

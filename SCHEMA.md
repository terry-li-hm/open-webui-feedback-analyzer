# Feedback Data Schema Documentation

This document describes the structure of feedback records exported from Open WebUI.

> **Note:** All UUIDs and identifiers in examples are fictional and for illustration purposes only.

## Overview

The export file is a JSON array containing feedback records. Each record captures user feedback on an AI assistant response, along with a complete snapshot of the conversation at the time the feedback was given.

## Root Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string (UUID) | Unique identifier for the feedback record |
| `user_id` | string (UUID) | Identifier of the user who submitted feedback |
| `version` | integer | Schema version number |
| `type` | string | Type of record, e.g., `"rating"` |
| `data` | object | Primary feedback data (see below) |
| `details` | object | Additional feedback details (see below) |
| `meta` | object | Metadata linking feedback to specific message |
| `base_models` | object | Mapping of model configuration to underlying model path |
| `snapshot` | object | Complete conversation state at time of feedback |
| `created_at` | integer | Unix timestamp (seconds) when feedback was created |
| `updated_at` | integer | Unix timestamp (seconds) when feedback was last updated |

## Data Object

Contains the primary feedback information.

| Field | Type | Description |
|-------|------|-------------|
| `rating` | integer | Thumbs up (`1`) or thumbs down (`-1`) |
| `model_id` | string | Identifier of the model configuration that generated the response |
| `sibling_model_ids` | array \| null | Alternative model IDs if multiple responses were generated |
| `reason` | string | Reason for the rating (see Reason Values below) |
| `comment` | string | Optional user comment explaining their feedback |
| `tags` | array | Optional tags associated with the feedback |

### Reason Values

Common values for the `reason` field:

- `accurate_information` - Information was accurate
- `inaccurate_information` - Information was inaccurate
- `helpful` - Response was helpful
- `not_helpful` - Response was not helpful

## Details Object

Contains additional rating information.

| Field | Type | Description |
|-------|------|-------------|
| `rating` | integer | Detailed rating on a 1-10 scale |

## Meta Object

Links the feedback to a specific message in the conversation.

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | string | Model configuration identifier |
| `message_id` | string (UUID) | ID of the specific message being rated |
| `message_index` | integer | Position of the message in the conversation |
| `chat_id` | string (UUID) | ID of the conversation |

## Base Models Object

Maps model configuration names to their underlying model paths.

```json
{
  "model-config-name": "/models/underlying-model-name"
}
```

## Snapshot Object

Contains the complete state of the conversation at the time feedback was given.

| Field | Type | Description |
|-------|------|-------------|
| `chat` | object | Chat metadata (see below) |
| `models` | array | List of model configurations available in this chat |
| `params` | object | Chat parameters |
| `history` | object | Conversation history (see below) |
| `tags` | array | Tags associated with the snapshot |
| `timestamp` | integer | Unix timestamp in **milliseconds** |
| `files` | array | Files attached to the conversation |
| `updated_at` | integer | Unix timestamp (seconds) |
| `created_at` | integer | Unix timestamp (seconds) |
| `share_id` | string \| null | Share link ID if conversation was shared |
| `archived` | boolean | Whether the conversation is archived |
| `pinned` | boolean | Whether the conversation is pinned |
| `meta` | object | Additional metadata |
| `folder_id` | string \| null | ID of folder containing this conversation |

### Snapshot.chat Object

| Field | Type | Description |
|-------|------|-------------|
| `id` | string (UUID) | Conversation identifier |
| `user_id` | string (UUID) | User identifier |
| `title` | string | Conversation title |

## History Object

Contains the conversation message tree.

| Field | Type | Description |
|-------|------|-------------|
| `messages` | object \| array | Messages indexed by ID or as array |
| `currentId` | string (UUID) | ID of the currently active message branch |
| `feedbackId` | string (UUID) | Links back to the feedback record ID |

### Message Object

Each message in the conversation has the following structure:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string (UUID) | Unique message identifier |
| `parentId` | string (UUID) \| null | Parent message ID (null for first message) |
| `childrenIds` | array | IDs of child messages (for branching conversations) |
| `role` | string | `"user"` or `"assistant"` |
| `content` | string | The message text |
| `timestamp` | integer | Unix timestamp (seconds) |
| `models` | array | (User messages) Available model configurations |
| `model` | string | (Assistant messages) Model configuration used |
| `modelName` | string | (Assistant messages) Human-readable model name |
| `modelIdx` | integer | (Assistant messages) Index of model in available models |
| `statusHistory` | array | (Assistant messages) RAG workflow steps |
| `done` | boolean | (Assistant messages) Whether response completed |
| `annotation` | object | (Assistant messages) Feedback annotation if rated |
| `feedbackId` | string (UUID) | (Assistant messages) Associated feedback record ID |
| `tags` | array | Tags on this message |
| `files` | array | Files attached to this message |

### StatusHistory Object (RAG Workflow)

Each assistant message may contain a `statusHistory` array documenting the retrieval-augmented generation workflow:

| Action | Fields | Description |
|--------|--------|-------------|
| `knowledge_search` | `query`, `done`, `hidden` | Initial knowledge base search |
| `queries_generated` | `queries` (array), `done` | Generated search query variations |
| `sources_retrieved` | `count`, `done` | Number of sources found |

Example:

```json
[
  {"action": "knowledge_search", "query": "user query here", "done": false},
  {"action": "queries_generated", "queries": ["query 1", "query 2", "query 3"], "done": false},
  {"action": "sources_retrieved", "count": 5, "done": true}
]
```

### Annotation Object

When a message is rated, it contains an annotation:

| Field | Type | Description |
|-------|------|-------------|
| `rating` | integer | Rating value (1 for thumbs up, -1 for thumbs down) |
| `tags` | array | Tags associated with this rating |
| `feedbackId` | string (UUID) | Links to the feedback record |
| `currentId` | string (UUID) | ID of the message being rated |

## Timestamp Formats

The data uses two timestamp formats:

| Format | Unit | Example | Used In |
|--------|------|---------|---------|
| 10 digits | Seconds | `1704067200` | Most `created_at`, `updated_at` fields |
| 13 digits | Milliseconds | `1704067200000` | `snapshot.timestamp` |

## Example Record Structure

```json
{
  "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
  "user_id": "11111111-2222-3333-4444-555555555555",
  "version": 0,
  "type": "rating",
  "data": {
    "rating": 1,
    "model_id": "my-model-config",
    "sibling_model_ids": null,
    "reason": "accurate_information",
    "comment": "",
    "tags": []
  },
  "details": {
    "rating": 10
  },
  "meta": {
    "model_id": "my-model-config",
    "message_id": "ffffffff-gggg-hhhh-iiii-jjjjjjjjjjjj",
    "message_index": 2,
    "chat_id": "kkkkkkkk-llll-mmmm-nnnn-oooooooooooo"
  },
  "base_models": {
    "my-model-config": "/models/Model-Name"
  },
  "snapshot": {
    "chat": {
      "id": "kkkkkkkk-llll-mmmm-nnnn-oooooooooooo",
      "user_id": "11111111-2222-3333-4444-555555555555",
      "title": "Example Conversation"
    },
    "models": ["my-model-config"],
    "history": {
      "messages": {},
      "currentId": "ffffffff-gggg-hhhh-iiii-jjjjjjjjjjjj"
    },
    "archived": false,
    "pinned": false
  },
  "created_at": 1704067200,
  "updated_at": 1704067200
}
```

# plan.md — Next Milestones (Group Chat → R2 Files → Pion SFU Video) + Deploy (Oracle Free Tier)

## Context
Current state:
- Base backend + auth + JWT.
- 1v1 chat implemented.

Next goal:
1) Add group chat end-to-end (API + WS + UI).
2) Add file sharing via Cloudflare R2 (direct upload with presigned URLs).
3) Add video calls using Pion SFU (signaling + room management + minimal UI).
4) Deploy to Oracle Always Free VM behind Caddy.

Guiding principles:
- Ship in vertical slices (backend + frontend + tests + minimal ops each stage).
- Prefer simple + correct over “perfect scale”.
- Keep secrets/config manual; everything else agent-driven.

---

## Stage 1 — Group Chat (complete MVP chat)

### Deliverables
- Create group, add/remove members, group title.
- Realtime group messages + history.
- Minimal group UX in web app.

### Tasks (Backend: REST)
- [Agent] Add/extend DB schema if needed:
  - `conversations` add: `type='group'`, `title`, `created_by`.
  - `conversation_members` with role (`admin`, `member`), joined_at.
- [Agent] Implement endpoints:
  - `POST /conversations` (type=group, title, initial_members[])
  - `GET /conversations` (list including groups + last message)
  - `GET /conversations/:id` (members + roles)
  - `POST /conversations/:id/members` (add by username)
  - `DELETE /conversations/:id/members/:userId`
  - `PATCH /conversations/:id` (rename group)
- [Agent] Add permission checks:
  - Only members can read/send.
  - Only admins can add/remove/rename (or allow members to add for MVP—choose one).
- [Agent] Add pagination for messages:
  - `GET /conversations/:id/messages?before=<cursor>&limit=50`

### Tasks (Backend: WebSocket)
- [Agent] Extend WS protocol:
  - Client → server: `room.join`, `room.leave`, `message.send`
  - Server → client: `message.new`, `room.member_added`, `room.member_removed`, `room.updated`
- [Agent] Implement fanout for group rooms:
  - Maintain `roomID -> sockets` mapping.
  - On `message.send`: validate membership, write DB, broadcast `message.new`.
- [Agent] Receipts (optional MVP-level):
  - Track `delivered` (when server broadcasts) and `read` (when client views room + sends receipt).

### Tasks (Frontend)
- [Agent] Add group create flow:
  - Modal: title + add members by username search.
- [Agent] Add members panel:
  - List members + role badge; add/remove UI.
- [Agent] Chat list UX:
  - Groups and DMs unified list with unread badge.
- [Agent] Room header:
  - Group title + member count + “Members” drawer.

### Acceptance checks
- Create a group with 3 users; all can see messages in realtime.
- Non-members cannot access messages (REST + WS).
- Basic pagination works.

---

## Stage 2 — File Sharing with Cloudflare R2 (presigned uploads)

### Why this design
Direct-to-R2 uploads avoid proxying bytes through your server and keep Oracle VM bandwidth/CPU low.
Presigned URLs are an S3 concept; Cloudflare R2 supports presigned URLs for PUT/GET up to 7 days expiry. [web:97]

### Deliverables
- Users can send files in chats.
- Files stored in R2; messages store attachment metadata.
- Download links work; basic size/type limits.

### Manual configuration (you)
1) Create Cloudflare R2 bucket.
2) Create R2 API credentials (Access Key ID + Secret Access Key).
3) Set env vars (server):
   - `R2_ACCOUNT_ID=...`
   - `R2_ACCESS_KEY_ID=...`
   - `R2_SECRET_ACCESS_KEY=...`
   - `R2_BUCKET=...`
   - `R2_PUBLIC_BASE_URL=` (optional; if you later add a public bucket/custom domain)
4) Decide max upload limits:
   - `MAX_UPLOAD_BYTES=...`
   - `ALLOWED_MIME_PREFIXES=image/,video/,application/pdf` (example)

Important note:
- R2 presigned URLs work with the S3 API domain and *cannot* be used with custom domains. [web:97]

### Tasks (Backend)
- [Agent] Add DB tables/fields:
  - `attachments`:
    - `id`, `uploader_id`, `bucket`, `object_key`, `mime`, `size_bytes`, `sha256`, `created_at`
  - `messages` add:
    - `attachment_id` nullable, OR `message_attachments` join table for multiple attachments
- [Agent] Implement API:
  - `POST /uploads/init`
    - Input: `conversation_id`, `filename`, `mime`, `size_bytes`
    - Validate membership + size/type
    - Generate `object_key` (e.g., `conv/<id>/<uuid>-<safe-filename>`)
    - Return: `attachment_id`, `object_key`, `presigned_put_url`, required headers
  - `POST /uploads/complete`
    - Input: `attachment_id`, `sha256` (optional)
    - Mark attachment ready (MVP can skip multipart/finalize)
  - `GET /attachments/:id/url`
    - Return presigned GET URL (short expiry)
- [Agent] Use AWS SigV4 signing via AWS SDK compatible client to generate presigned URLs for R2. [web:97]
- [Agent] Extend message send:
  - Allow `message.send` with `attachment_id` and optional caption text.

### Tasks (Frontend)
- [Agent] Composer attachments:
  - “Attach” button + drag/drop
  - Client calls `/uploads/init` → PUT to presigned URL → `/uploads/complete` → send message with `attachment_id`
- [Agent] Message rendering:
  - Image preview thumbnails
  - Other files show filename + size; click to fetch `/attachments/:id/url` and open
- [Agent] UX details:
  - Upload progress bar
  - Failure UI with retry

### Acceptance checks
- Upload an image and see it in group chat.
- Attachment download works for other members.
- Unauthorized user cannot init upload for a room they’re not in.

---

## Stage 3 — Video Calls with Pion SFU (room calls + placeholders)

### Goal (MVP)
- “Join call” in a group room.
- Basic SFU forwarding (no recordings, no fancy layouts).
- Minimal UI: grid of participants + mute/cam toggles.

### Architectural choice
- Keep SFU as a separate Go service/process (`sfu/`) from chat API, but share the same auth/JWT.
- Use WebSocket signaling: browser connects to SFU signaling WS, exchanges SDP/ICE.

Reference starting point:
- Pion provides an SFU WebSocket example (`sfu-ws`) showing many-to-many SFU over WS. [web:102]

### Manual configuration placeholders (you)
1) Decide call domain/subdomain:
   - `CALLS_BASE_URL=https://calls.<domain>`
2) If you want reliable NAT traversal, plan a TURN server:
   - `TURN_URL=turn:<domain>:3478`
   - `TURN_USERNAME=...`
   - `TURN_PASSWORD=...`
   Coturn supports long-term credentials (username/key) configuration. [web:106]
3) Open ports on Oracle VM / firewall:
   - 80/443 (HTTP/HTTPS)
   - UDP port range for WebRTC (e.g., 10000-20000) (exact range depends on SFU config)
4) Set env vars for SFU:
   - `JWT_PUBLIC_KEY=...` or shared `JWT_SIGNING_KEY`
   - `ICE_STUN_URLS=stun:stun.l.google.com:19302` (example placeholder)
   - `ICE_TURN_URLS=...` (optional)
   - `SFU_UDP_PORT_RANGE=10000-20000`

### Tasks (SFU backend service)
- [Agent] Create `sfu/` Go module/service:
  - WebSocket signaling endpoint: `/ws`
  - Room model: `room_id` == conversation_id (simplest mapping)
- [Agent] Auth:
  - Require JWT on WS connect; extract `user_id`.
- [Agent] Implement join/leave:
  - Client sends `join { room_id }`
  - Server returns participant list + SFU params
- [Agent] Implement WebRTC:
  - For each participant: create PeerConnection, handle SDP offer/answer, ICE candidates
  - Forward tracks to other participants (SFU behavior)
- [Agent] Add basic controls:
  - Mute/unmute audio (client-side track enabled/disabled)
  - Camera on/off
- [Agent] Metrics/logging:
  - Log room join/leave, active rooms, participants count

### Tasks (Chat backend integration)
- [Agent] Add “call state” endpoints (minimal):
  - `POST /conversations/:id/call/start` (optional; can be implicit when first joins)
  - `POST /conversations/:id/call/end`
  - Or just treat calls as ephemeral (no DB) for MVP.
- [Agent] WebSocket event to room:
  - `call.started`, `call.ended`, `call.participants_updated`

### Tasks (Frontend: calls UI)
- [Agent] In group header: “Join call” button.
- [Agent] Call screen:
  - Local preview tile + remote tiles grid
  - Controls row: mic, cam, leave
  - Show participant names
- [Agent] Network UX:
  - “Connecting…” state
  - Reconnect on WS disconnect (simple)

### Acceptance checks
- 2 users in same group can join and see/hear each other.
- 3+ users can join (basic SFU forwarding).
- No auth bypass: only group members can join that room’s call.

---

## Stage 4 — Deploy to Oracle Always Free (single VM) with Caddy

### Deployment topology
- Caddy (reverse proxy + TLS)
- Backend API (Go) on localhost:8080
- Web frontend (static build) served by Caddy OR separate Node build artifact
- SFU signaling service on localhost:8090 (plus UDP ports for media)

Caddy HTTPS:
- Use Caddy automatic HTTPS for certificates and redirects. [web:67]

### Manual steps (you)
1) Provision Oracle Always Free VM, attach domain DNS A records:
   - `<domain>` → VM public IP
   - `api.<domain>` → VM public IP
   - `calls.<domain>` → VM public IP
2) Open firewall ports:
   - 80/tcp, 443/tcp
   - SFU UDP range you selected
   - (Optional) 3478/tcp+udp for coturn if hosted on same VM
3) Install:
   - Docker + docker compose OR systemd (choose one approach)
4) Set secrets:
   - OAuth creds, JWT keys
   - Postgres creds
   - R2 creds
   - TURN creds (if used)

### Agent tasks (infra code)
- [Agent] Create `infra/` artifacts:
  - `Caddyfile` routing:
    - `api.<domain>` → `localhost:8080`
    - `calls.<domain>` → `localhost:8090` (WS + HTTPS)
    - `<domain>` serves static web build
  - `docker-compose.prod.yml` (if Docker approach)
  - `systemd` unit files (if systemd approach)
  - `deploy.sh`:
    - build backend + sfu
    - build web
    - upload artifacts to VM (rsync/scp)
    - restart services
- [Agent] Add health checks:
  - `/healthz` on api and sfu
- [Agent] Add production config docs:
  - `.env.example` for api/sfu
  - “how to rotate JWT signing key”
  - “how to revoke refresh tokens” (if implemented)

### Acceptance checks
- `https://<domain>` loads web app.
- `https://api.<domain>/healthz` OK.
- Group chat + file share works in production.
- `https://calls.<domain>` signaling reachable; calls work (with TURN if required).

---


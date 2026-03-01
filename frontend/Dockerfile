# ───────────────────────────────────────────────────────────────────────────
# Frontend Dockerfile — PLACEHOLDER
# ───────────────────────────────────────────────────────────────────────────
# This file is a placeholder for the Frontend Team.
# Replace the entire contents with the real build steps for your chosen
# framework once the frontend is implemented.
#
# ── Option A: Plain HTML / JS (static files served by nginx) ───────────────
#
#   FROM nginx:alpine
#   COPY . /usr/share/nginx/html
#   EXPOSE 80
#
# ── Option B: React (Vite) ─────────────────────────────────────────────────
#
#   FROM node:20-alpine AS builder
#   WORKDIR /app
#   COPY package*.json ./
#   RUN npm ci
#   COPY . .
#   RUN npm run build          # outputs to /app/dist
#
#   FROM nginx:alpine
#   COPY --from=builder /app/dist /usr/share/nginx/html
#   EXPOSE 80
#
# ── Option C: React (Create React App) ────────────────────────────────────
#
#   FROM node:20-alpine AS builder
#   WORKDIR /app
#   COPY package*.json ./
#   RUN npm ci
#   COPY . .
#   RUN npm run build          # outputs to /app/build
#
#   FROM nginx:alpine
#   COPY --from=builder /app/build /usr/share/nginx/html
#   EXPOSE 80
#
# ── WebSocket proxy note ───────────────────────────────────────────────────
# If you serve the frontend from nginx and the backend is on a different
# origin, add a reverse-proxy location block to nginx.conf so the browser
# can reach the WebSocket endpoint without CORS issues:
#
#   location /ws/ {
#       proxy_pass http://backend:8000;
#       proxy_http_version 1.1;
#       proxy_set_header Upgrade $http_upgrade;
#       proxy_set_header Connection "upgrade";
#   }
# ───────────────────────────────────────────────────────────────────────────

FROM nginx:alpine

# Serve a placeholder page until the frontend is implemented
RUN printf '<html>\n<head><title>Frontend Placeholder</title></head>\n<body>\n<h1>Frontend not yet implemented</h1>\n<p>Replace <code>frontend/Dockerfile</code> with the real frontend build.</p>\n</body>\n</html>\n' \
    > /usr/share/nginx/html/index.html

EXPOSE 80

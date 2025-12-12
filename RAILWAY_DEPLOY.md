# Railway Deployment Guide

## Prerequisites
- Railway account (railway.app)
- Git repo pushed to GitHub/GitLab

## Quick Start

1. **Create a new Railway project**
   - Go to [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub Repo"
   - Select your flowtype repository

2. **Add PostgreSQL service**
   - Click "+ Add" → Select "PostgreSQL"
   - Railway auto-provisions a Postgres instance

3. **Configure environment variables**
   - In your project settings, add:
     ```
     DATABASE_URL=${{ Postgres.DATABASE_URL }}
     PYTHONUNBUFFERED=1
     VITE_API_URL=https://<your-railway-domain>/api
     UMAMI_WEBSITE_ID=<your-umami-id>
     ```

4. **Deploy**
   - Push to main branch or manually trigger deploy
   - Railway auto-detects the `Dockerfile` and builds

5. **Set up Umami Analytics (optional)**
   - Host Umami at umami.is or self-host
   - Get your website ID and update `UMAMI_WEBSITE_ID`
   - Update `data-website-id` in `frontend/index.html`

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `DATABASE_URL` | PostgreSQL connection | `postgresql://user:pass@host/db` |
| `PYTHONUNBUFFERED` | Python logging | `1` |
| `VITE_API_URL` | Frontend API endpoint | `https://flowtype.railway.app/api` |
| `UMAMI_WEBSITE_ID` | Analytics tracking ID | `your-id-here` |

## Monitoring

- View logs: Railway dashboard → Logs tab
- Check health: Visit `https://<your-domain>/` 
- Database: Connect with any PostgreSQL client to `$DATABASE_URL`

## Local Testing

```bash
docker compose up --build
# Access at https://localhost
```

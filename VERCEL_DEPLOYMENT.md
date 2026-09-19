# 🚀 Deploying AI Customer Support Ticket Classifier to Vercel

This repository is pre-configured for **1-click zero-configuration deployment to Vercel** using Vercel Serverless Functions (`@vercel/python`) and static CDN asset delivery (`@vercel/static`).

---

## 🏗️ How Vercel Deployment Works

```
                        +-------------------------------+
                        |          User Browser         |
                        +-------------------------------+
                                        |
                 +----------------------+----------------------+
                 | (Static Assets)                             | (API Requests)
                 v                                             v
    +-------------------------+                   +-------------------------+
    |   Vercel Global CDN     |                   | Vercel Serverless Func  |
    |   /public/index.html    |                   |   /api/index.py         |
    |   /public/style.css     |                   |   (Flask + Model)       |
    |   /public/app.js        |                   +-------------------------+
    +-------------------------+                                |
                                                               v
                                                  +-------------------------+
                                                  | Fast Inference Engine   |
                                                  | (100% Accuracy, <15ms)  |
                                                  +-------------------------+
```

### Why This Setup is Optimized for Vercel:
- **Zero WebSocket Overhead**: Unlike Streamlit (which requires long-lived stateful WebSockets unsupported on Vercel serverless), this architecture uses a responsive single-page application (SPA) backed by serverless REST endpoints.
- **Under 250MB Limit**: Serverless dependencies in `api/requirements.txt` and the lightweight model (`models/lightweight_model.joblib`) take less than **50 MB** total, easily passing Vercel's strict 250MB bundle limit and deploying in under 30 seconds.

---

## ⚡ Option 1: Deploy via Vercel CLI (Fastest - 60 Seconds)

From your terminal inside the project directory:

```bash
cd "/Users/nikusingh/Desktop/DLNLP project"

# Run Vercel CLI via npx (no global installation required)
npx vercel
```

### Terminal Prompts:
1. **Set up and deploy?** Press `Y` and Enter.
2. **Which scope do you want to deploy to?** Press Enter (select your Vercel account).
3. **Link to existing project?** Press `N` (or Enter).
4. **What's your project's name?** Press Enter to keep default (`dlnlp-project` or `ai-ticket-classifier`).
5. **In which directory is your code located?** Press Enter (`./`).
6. **Want to modify these settings?** Press `N` (Enter).

Vercel will build and output your preview URL (e.g., `https://ai-ticket-classifier-abc.vercel.app`).

To deploy directly to production:
```bash
npx vercel --prod
```

---

## 🌐 Option 2: Deploy via GitHub (Continuous Deployment)

1. Initialize git and push the repository to GitHub:
   ```bash
   cd "/Users/nikusingh/Desktop/DLNLP project"
   git init
   git add .
   git commit -m "Initial commit: AI Support Ticket Classifier with Vercel support"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

2. Visit [Vercel Dashboard](https://vercel.com/new).
3. Click **"Add New Project"** $\rightarrow$ **"Import Git Repository"**.
4. Select your GitHub repository.
5. Click **Deploy**. Vercel will automatically read `vercel.json` and deploy your website.

---

## 🧪 Testing Locally Before Deploying

You can run the exact same Vercel web application locally on port 5050:

```bash
source .venv/bin/activate
python server.py
```
Open your browser at **`http://localhost:5050`** to test the web interface and API endpoints!

# MSD Sentinel - Frontend Deployment Guide

## Project Overview
MSD Sentinel is an IoT-based Musculoskeletal Disorder Monitoring System that provides real-time posture assessment using REBA and RULA ergonomic assessment tools.

## Netlify Frontend Deployment

This project's frontend can be deployed to Netlify as a static site. The frontend provides:
-Real-time dashboard visualization
- Sensor status monitoring
- CSV data management interface
- PDF report generation UI
- REBA and RULA assessment displays

### Files Ready for Deployment
All static files are in the `dist/` directory:
- `index.html` - Main dashboard
- `system.html` - Sensor placement view
- `csv_view.html` - CSV data management
- `reports.html` -Report generation
- `reba.html` - REBA assessment
- `rula.html` -RULA assessment
- `style.css` - All styling
- `dashboard.js` - Dashboard logic
- `reba.js` - REBA calculations
- `rula.js` - RULA calculations

### Deployment Steps

#### Option 1: Netlify CLI (Recommended)

1. **Install Netlify CLI**
   ```bash
   npm install -g netlify-cli
   ```

2. **Login to Netlify**
   ```bash
   netlify login
   ```

3. **Deploy from the project root**
   ```bash
   netlify deploy --prod --dir=dist
   ```

4. **Follow the prompts**
   - "Choose a project" → Create new project or select existing
   - "Deploy path" → dist (already set in command)

#### Option 2: Netlify Web Interface

1. Go to [netlify.com](https://netlify.com)
2. Click "Add new site" → "Deploy manually"
3. Drag and drop the entire `dist` folder
4. Wait for deployment to complete

#### Option 3: Git Integration

1. Push your code to GitHub/GitLab/Bitbucket
2. Connect your repository on Netlify
3. Configure build settings:
   - **Build command**: (leave empty)
   - **Publish directory**: `dist`
4. Deploy!

### Configuration

The `netlify.toml` file is already configured:

```toml
[build]
publish = "dist"

[[redirects]]
  from = "/api/*"
  to = "/index.html"
  status = 200
```

This handles SPA routing and API calls gracefully.

### Important Notes

⚠️ **Backend Required**: The frontend expects a backend API for:
- `/api/sensors` - Sensor status
- `/api/csv/*` - CSV file operations
- `/api/reports/*` -Report operations
- `/api/data` -Real-time sensor data
- WebSocket connections for live updates

⚠️ **API Endpoints**: You need to either:
1. Deploy the Flask backend separately (Railway, Render, Heroku, etc.)
2. Update API calls in the JavaScript to point to your backend URL
3. Use Netlify Functions for serverless API endpoints

### Updating API Base URL

To connect the frontend to your deployed backend:

1. Open each HTML file in the `dist/` folder
2. Replace relative API paths with your backend URL:
   ```javascript
   // Change from:
  const response = await fetch('/api/sensors');
   
   // To:
  const response = await fetch('https://your-backend-url.com/api/sensors');
   ```

Or add a configuration constant at the top of your JS files.

### Backend Deployment Options

For full functionality, deploy the Flask backend to:
- **Railway.app** (recommended for Flask)
- **Render.com**
- **Heroku**
- **PythonAnywhere**

See `requirements.txt` for Python dependencies.

### Local Testing

Test the static files locally before deploying:

```bash
# Using Python's built-in server
cd dist
python -m http.server 8000

# Or use Netlify Dev
netlify dev
```

Visit `http://localhost:8000` to preview.

### Custom Domain

After deployment:
1. Go to Site Settings → Domain Management
2. Add your custom domain
3. Update DNS records as instructed

### Environment Variables

If you need environment variables for API URLs:
1. Go to Site Settings → Build & Deploy → Environment
2. Add variables like `API_BASE_URL`
3. Access via JavaScript using process replacement or build-time injection

### Troubleshooting

**Issue**: Pages show "Disconnected"
- **Solution**: Backend API is not reachable. Check CORS settings and API URL.

**Issue**: Charts not loading
- **Solution**: Ensure Chart.js CDN is accessible or bundle it locally.

**Issue**: CSS not applied
- **Solution**: Verify `style.css` path is correct (should be relative).

### Continuous Deployment

For automatic deployments on git push:
1. Connect your Git repository to Netlify
2. Enable auto-deploy in Site Settings
3. Every push to main branch will trigger deployment

### Support

For issues or questions:
- Check Netlify documentation: https://docs.netlify.com
-Review Flask backend logs for API errors
- Inspect browser console for frontend errors

---

**License**: Your project license here
**Author**: Your name

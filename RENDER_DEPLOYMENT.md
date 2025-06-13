# 🎛️ AI Mixer Tournament - Render Deployment Guide

## ⚠️ Important: File Inclusion/Exclusion Strategy

The `.gitignore` file has been carefully configured to:

1. **EXCLUDE all data files** except:
   - `data/README.md` - Instructions for data setup
   - `data/targets_example.json` - Small example file

2. **INCLUDE all model deployment files**:
   - The models in `models/deployment/` are all small enough (<50MB)
   - Total size is approximately 72MB for all models
   - These are required for the application to function

3. **EXCLUDE large binary files**:
   - `.npy` - NumPy array files
   - `.wav`, `.mp3`, etc. - Audio files
   - Larger model checkpoints

Before committing, double-check with:
```bash
# See what will be added to git
git add .
git status
```

Look for any unwanted large files in the "Changes to be committed" section.

## Model Deployment Strategy

After running `prepare_models_for_deployment.py`, we've confirmed that all 6 models are under 50MB:
- baseline_cnn (4.88 MB)
- enhanced_cnn (11.66 MB)
- improved_baseline_cnn (4.88 MB)
- improved_enhanced_cnn (3.08 MB)
- retrained_enhanced_cnn (11.66 MB)
- weighted_ensemble (36.16 MB)

These models have been copied to the `models/deployment/` directory, which should be included in your Git repository. Since all models are relatively small, they can be deployed directly with your application without needing to implement on-demand downloading.

### Specific Environment Setup for Models

When deploying to Render, make sure to use this environment variable:
```
MODELS_DIR=../models/deployment
```

This ensures your application loads the optimized models from the deployment directory.

## 🚀 Free Hosting with Render.com

This guide explains how to deploy your AI Mixer Tournament application for free on Render.com, which allows model inference and online training functionality without requiring payment information.

## Preparing Your Git Repository for Deployment

Before committing to Git and deploying to Render, make sure your repository doesn't include large data files:

1. **The `.gitignore` file** has been configured to exclude:
   - The entire `/data/` directory (except for README.md and a few small JSON examples)
   - Large model files (*.pth)
   - Audio files (*.wav, *.mp3, etc.)
   - Generated features and spectrograms

2. **To commit your code for deployment:**
   ```bash
   # Check what files will be committed
   git status

   # Add all files except those in .gitignore
   git add .

   # Commit with a descriptive message
   git commit -m "Prepare for Render deployment"

   # Push to your repository
   git push
   ```

3. **Alternative model handling options:**
   - Store smaller models in the repository
   - Use a model hub like Hugging Face for larger models
   - Set up cloud storage (S3, Google Cloud Storage) for on-demand downloading

## Repository Organization for Deployment

Based on your current repository status, here's how to organize your commit for deployment:

### Files to Include
All the new files should be included, especially:
- `.gitignore` - Ensures data files aren't committed
- `RENDER_DEPLOYMENT.md` - This deployment guide
- `render.yaml` & `render/` - Render configuration
- `models/deployment/` - Your optimized models
- `tournament_webapp/` - The core application
- `src/enhanced_musical_intelligence.py` & `src/production_ai_mixer.py` - New production-ready components

### Files Being Removed
Several documentation files are being deleted. If these contain valuable information, consider:
1. Consolidating them into fewer, more organized documents
2. Moving critical content to README files in relevant directories

### Commit Process
```bash
# Add all new files and changes
git add .

# Check what will be committed (verify data files are excluded)
git status

# Commit with a descriptive message
git commit -m "Prepare for Render deployment with optimized models"

# Push to your repository
git push origin main
```

## Why Render?

Render offers several advantages:
- **Free Tier**: No credit card required
- **GitHub Integration**: Deploy directly from your repository
- **Auto-Deploy**: Automatic deployment on code changes
- **Environment Variables**: Secure configuration management
- **Scaling Options**: Easy upgrade path when you need more resources

## Deployment Steps

### 1. Create a Render Account

Sign up at [Render.com](https://render.com) (no credit card required).

### 2. Create a New Web Service

1. From your Render dashboard, click **New** and select **Web Service**
2. Connect your GitHub repository (or upload your code)
3. Configure your service:
   - **Name**: `ai-mixer-tournament`
   - **Runtime**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `cd tournament_webapp && python dev_server.py --production`

### 3. Configure Environment Variables

Add these environment variables in the Render dashboard:
- `PYTHON_VERSION`: `3.10.12`
- `PRODUCTION`: `true`
- `PORT`: `10000` (Render will provide a `PORT` variable automatically)
- `HOST`: `0.0.0.0`
- `ALLOWED_ORIGINS`: `https://your-app.onrender.com,http://localhost:3000`
- `MODELS_DIR`: `../models`

### 4. Deploy the Service

Click **Create Web Service** and Render will start the deployment process.

## Handling Model Files

Render's free tier has limited disk space (about 1GB), so consider these options for your model files:

1. **Include Small Models in Repository**: If models are small enough
2. **Use External Storage**: Configure your app to download models from a service like AWS S3, Google Cloud Storage, or Hugging Face Hub
3. **On-Demand Loading**: Load models only when needed and release memory when inactive

## Application Structure for Render

```
mixer/
├── requirements.txt         # Python dependencies
├── runtime.txt              # Python version
├── render/
│   └── render.yaml          # Render configuration
├── tournament_webapp/
│   ├── backend/             # FastAPI application
│   ├── frontend/            # React frontend
│   └── dev_server.py        # Development/production server
└── models/                  # ML models
```

## Resource Management for Training

For model training on Render's free tier:

1. **Optimize Training Code**: Use efficient algorithms and batch sizes
2. **Implement Checkpointing**: Save progress regularly
3. **Consider Lightweight Models**: Smaller architectures with fewer parameters
4. **Use Transfer Learning**: Start from pre-trained models to reduce training time
5. **Implement Early Stopping**: Avoid unnecessary computation

## Monitoring Your Application

Render provides logs and metrics for monitoring your application:
- **Logs**: Real-time logs for debugging
- **Metrics**: CPU, memory, and network usage
- **Health Checks**: Automatically restart if your service fails

## Scaling Up When Needed

When your project needs more resources:
1. Upgrade to Render's paid tier
2. Use the same configuration with more resources
3. No migration needed - just change the plan

## Troubleshooting

Common issues and solutions:
- **Deployment Failures**: Check build logs for errors
- **Application Crashes**: Review application logs
- **Performance Issues**: Consider optimizing your code or upgrading your plan

## Additional Resources

- [Render Documentation](https://render.com/docs)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Optimizing Python for Production](https://render.com/docs/python)

---

Remember that while Render's free tier is great for development and small-scale projects, intensive model training might require upgrading to a paid tier or using specialized ML platforms.

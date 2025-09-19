# CORS Configuration Guide

This guide explains how to configure CORS (Cross-Origin Resource Sharing) for both development and production environments.

## Backend Configuration

The backend uses environment variables to configure CORS origins dynamically:

### Environment Variables

Create a `.env` file in the `backend/` directory with:

```env
# CORS Origins - Comma separated list of allowed origins
CORS_ORIGINS="http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"
```

### Different Environment Configurations

#### Development

```env
CORS_ORIGINS="*"
# or specific local origins:
CORS_ORIGINS="http://localhost:3000,http://localhost:5173"
```

#### Production (Vercel + AWS)

```env
CORS_ORIGINS="https://your-app.vercel.app,https://your-custom-domain.com"
```

#### Production (Multiple domains)

```env
CORS_ORIGINS="https://domain1.com,https://domain2.com,https://subdomain.domain.com"
```

## Frontend Configuration

The frontend uses environment variables to configure the backend API URL:

### Environment Variables

Create a `.env` file in the `frontend/` directory:

```env
# Backend API URL
VITE_BACKEND_API="http://localhost:8000"
```

### Different Environment Configurations

#### Development

```env
VITE_BACKEND_API="http://localhost:8000"
```

#### Production (AWS)

```env
VITE_BACKEND_API="https://your-aws-api-domain.com"
```

#### Production (GCP)

```env
VITE_BACKEND_API="https://api-lopt-540193079740.us-central1.run.app"
```

## Deployment-Specific Instructions

### Vercel Deployment

1. In your Vercel project settings, add these environment variables:

   ```
   VITE_BACKEND_API=https://your-backend-url.com
   ```

2. For the backend, make sure to update CORS_ORIGINS to include your Vercel domain:
   ```
   CORS_ORIGINS=https://your-app.vercel.app
   ```

### AWS Deployment

1. Set environment variables in your AWS deployment configuration
2. Update both frontend and backend environment variables accordingly

## Troubleshooting CORS Issues

1. **Check Origins**: Ensure your frontend domain is listed in CORS_ORIGINS
2. **Check Protocols**: Match http/https protocols correctly
3. **Check Ports**: Include port numbers for local development
4. **Check Subdomains**: Add all subdomains you're using

## Development Tips

- Use `CORS_ORIGINS="*"` for development only
- Always specify exact origins in production
- Test CORS configuration after any domain changes
- Check browser developer tools for CORS error details

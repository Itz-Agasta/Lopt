FROM node:18-alpine

WORKDIR /app

# Install dependencies first (for better caching)
COPY package.json package-lock.json ./
RUN npm ci --quiet

# Copy source code
COPY . .

# Expose port for vite dev server
EXPOSE 5173

# Start development server
CMD ["npm", "run", "dev", "--", "--host"]
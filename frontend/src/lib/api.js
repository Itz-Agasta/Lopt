import axios from "axios";

// Get API URL from environment variables
const getApiUrl = () => {
    // Use Vite environment variable, fallback to production URL if not set
    return import.meta.env.VITE_BACKEND_API || "https://api-lopt-540193079740.us-central1.run.app";
};

export async function analyzeFile(file){
    const formData = new FormData();
    formData.append('file', file);

    const API_URL = getApiUrl();
    console.log('Using API URL:', API_URL);
    
    const response = await axios.post(`${API_URL}/playground/`, formData, {
        headers: {
            "Content-Type": "multipart/form-data",
        },
    });
    
    return response.data;
}

export async function try_sample(idx, type){
    const API_URL = getApiUrl();
    
    // Import samples to get the actual file URLs
    const { default: samples } = await import('../components/static/samples.js');
    
    if (idx < 0 || idx >= samples.length) {
        throw new Error(`Invalid sample index: ${idx}`);
    }
    
    const sample = samples[idx];
    
    try {
        // Fetch the sample file using its source URL
        const fileResponse = await fetch(sample.source);
        if (!fileResponse.ok) {
            throw new Error(`Failed to fetch sample file: ${sample.name}`);
        }
        
        const blob = await fileResponse.blob();
        const filename = `${sample.name}.${sample.source.includes('.mp4') ? 'mp4' : 'jpg'}`;
        
        // Create FormData with the actual file
        const formData = new FormData();
        formData.append('file', blob, filename);
        
        console.log('Sending sample file:', filename);
        
        const response = await axios.post(`${API_URL}/playground/test`, formData, {
            headers: {
                "Content-Type": "multipart/form-data",
            },
        });
        
        return response.data;
    } catch (error) {
        console.error('Error with sample file:', error);
        throw error;
    }
}


import axios from "axios";

export async function analyzeFile(file){
    const formData = new FormData();
    formData.append('file', file);

    const API_URL = import.meta.env.VITE_BACKEND_API;
    //console.log(API_URL);
    const response = await axios.post(`${API_URL}/playground/`, formData, {
        headers: {
            "Content-Type": "multipart/form-data",
        },
    });
    
    return response.data;
}

export async function try_sample(idx, type){

    const API_URL = import.meta.env.VITE_BACKEND_API;
    
    const response = await axios.post(`${API_URL}/playground/test`, {
        pos: idx,
        type: type
    });
    
    return response.data;
}


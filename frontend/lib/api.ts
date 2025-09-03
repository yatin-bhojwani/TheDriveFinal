import type { FileSystemItem, FolderResponse, SearchFilters } from '@/types';

// API for file management and authentication
const API_URL = process.env.NEXT_PUBLIC_API_URL 

// API for the RAG pipeline chat service
const RAG_API_URL = 'http://34.131.37.148:8080';

async function fetchApi(endpoint: string, options: RequestInit = {}) {
    const headers = new Headers(options.headers);

    if (!(options.body instanceof FormData)) {
        headers.set('Content-Type', 'application/json');
    }

    const response = await fetch(`${API_URL}${endpoint}`, {
        ...options,
        headers,
        credentials: 'include'
    });

    if (response.status === 401) {
        if (typeof window !== 'undefined' && window.location.pathname !== '/login') {
            window.location.href = '/login';
        }
        throw new Error('Unauthorized');
    }

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'An API error occurred');
    }

    if (response.status === 204) {
        return null;
    }

    return response.json();
}

export const api = {
    // --- TheDrive Backend Calls (Port 8000) ---
    login: (formData: FormData): Promise<{ message: string }> => {
        return fetchApi('/auth/login', {
            method: 'POST',
            body: formData
        });
    },
    signup: (email: string, password: string): Promise<{ message: string }> => {
        return fetch(`${API_URL}/auth/signup`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
        }).then(async res => {
            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Signup failed');
            }
            return res.json();
        });
    },
    logout: (): Promise<{ message: string }> => {
        return fetchApi('/auth/logout', { method: 'POST' });
    },
    getMe: (): Promise<{ email: string, id: number }> => {
        return fetchApi('/auth/me');
    },
    getItems: (parentId: string): Promise<FolderResponse> => {
        return fetchApi(`/drive/items?parentId=${parentId}`);
    },
    createFolder: (name: string, parentId: string): Promise<FileSystemItem> => {
        return fetchApi('/drive/folder', {
            method: 'POST',
            body: JSON.stringify({ name, parentId }),
        });
    },
    renameItem: (itemId: string, newName: string): Promise<FileSystemItem> => {
        return fetchApi(`/drive/item/${itemId}`, {
            method: 'PUT',
            body: JSON.stringify({ name: newName }),
        });
    },
    deleteItem: (itemId: string): Promise<null> => {
        return fetchApi(`/drive/item/${itemId}`, { method: 'DELETE' });
    },
    uploadFile: (file: File, parentId: string): Promise<FileSystemItem> => {
        const formData = new FormData();
        formData.append('file', file);
        return fetchApi(`/drive/upload?parentId=${parentId}`, {
            method: 'POST',
            body: formData,
        });
    },
    getViewLink: (itemId: string): Promise<{ url: string }> => {
        return fetchApi(`/drive/item/${itemId}/view-link`);
    },
    getIngestionStatus: (itemId: string): Promise<{ id: string; ingestion_status: string }> => {
        return fetchApi(`/drive/item/${itemId}/ingestion-status`);
    },
    searchItems: (filters: SearchFilters): Promise<FileSystemItem[]> => {
        return fetchApi('/drive/search', {
            method: 'POST',
            body: JSON.stringify(filters),
        });
    },

    // RAG Pipeline SSE Call (Port 8080)
    getChatStream: (
        query: string,
        sessionId: string,
        context: { fileId?: string | null; folderId?: string | null }
    ): EventSource => {
    let scope: 'drive' | 'folder' | 'file' = 'drive';
    const params = new URLSearchParams({ query });

    if (context.fileId) {
        scope = 'file';
        params.append('file_id', context.fileId);
    } else if (context.folderId && context.folderId !== 'root') {
        scope = 'folder';
        params.append('folder_id', context.folderId);
    }
    
    params.append('scope', scope);

    // This targets the endpoint from your `drv` RAG service
    const url = `${RAG_API_URL}/chat/conversation/${sessionId}/stream?${params.toString()}`;
    
    return new EventSource(url);
},
};
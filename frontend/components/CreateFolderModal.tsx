'use client';

import { useState, useEffect } from 'react';
import { X, FolderPlus } from 'lucide-react';

interface CreateFolderModalProps {
    isOpen: boolean;
    onClose: () => void;
    onCreate: (folderName: string) => void;
    isNameTaken: (folderName: string) => boolean;
}

export const CreateFolderModal = ({ isOpen, onClose, onCreate, isNameTaken }: CreateFolderModalProps) => {
    const [folderName, setFolderName] = useState('');
    const [error, setError] = useState<string | null>(null);

    // Reset state when modal opens/closes
    useEffect(() => {
        if (!isOpen) {
            setFolderName('');
            setError(null);
        }
    }, [isOpen]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        const trimmedName = folderName.trim();
        if (!trimmedName) return;

        if (isNameTaken(trimmedName)) {
            setError('A folder with this name already exists.');
            return;
        }
        
        onCreate(trimmedName);
    };

    const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFolderName(e.target.value);
        if (error) {
            setError(null);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 w-full max-w-md relative">
                <button onClick={onClose} className="absolute top-3 right-3 p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700">
                    <X className="h-5 w-5" />
                </button>
                <h2 className="text-lg font-semibold mb-4 flex items-center gap-2"><FolderPlus /> New Folder</h2>
                <form onSubmit={handleSubmit}>
                    <input
                        type="text"
                        value={folderName}
                        onChange={handleNameChange}
                        placeholder="Untitled folder"
                        className={`w-full px-3 py-2 border rounded-md bg-transparent focus:outline-none focus:ring-2 ${error ? 'border-red-500 focus:ring-red-500' : 'border-gray-300 dark:border-gray-600 focus:ring-blue-500'}`}
                        autoFocus
                    />
                    {error && <p className="text-red-500 text-sm mt-2">{error}</p>}
                    <div className="flex justify-end gap-3 mt-6">
                        <button type="button" onClick={onClose} className="px-4 py-2 rounded-md text-sm font-medium hover:bg-gray-100 dark:hover:bg-gray-700">Cancel</button>
                        <button type="submit" className="px-4 py-2 rounded-md text-sm font-medium bg-blue-600 text-white hover:bg-blue-700 disabled:bg-blue-400" disabled={!folderName.trim()}>Create</button>
                    </div>
                </form>
            </div>
        </div>
    );
};
'use client';

import { useState, useEffect } from 'react';
import { X, Edit } from 'lucide-react';
import type { FileSystemItem } from '@/types';

interface RenameModalProps {
    item: FileSystemItem | null;
    onClose: () => void;
    onRename: (newName: string) => void;
    isNameTaken: (name: string, id: string) => boolean;
}

export const RenameModal = ({ item, onClose, onRename, isNameTaken }: RenameModalProps) => {
    const [newName, setNewName] = useState('');
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (item) {
            setNewName(item.name);
            setError(null);
        }
    }, [item]);

    if (!item) return null;

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        const trimmedName = newName.trim();
        if (!trimmedName) return;

        if (trimmedName !== item.name && isNameTaken(trimmedName, item.id)) {
            setError(`A ${item.type} with this name already exists.`);
            return;
        }
        
        onRename(trimmedName);
    };

    const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setNewName(e.target.value);
        if (error) setError(null);
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 w-full max-w-md relative">
                <button onClick={onClose} className="absolute top-3 right-3 p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700">
                    <X className="h-5 w-5" />
                </button>
                <h2 className="text-lg font-semibold mb-4 flex items-center gap-2"><Edit /> Rename</h2>
                <form onSubmit={handleSubmit}>
                    <input
                        type="text"
                        value={newName}
                        onChange={handleNameChange}
                        className={`w-full px-3 py-2 border rounded-md bg-transparent focus:outline-none focus:ring-2 ${error ? 'border-red-500 focus:ring-red-500' : 'border-gray-300 dark:border-gray-600 focus:ring-blue-500'}`}
                        autoFocus
                        onFocus={(e) => e.target.select()}
                    />
                    {error && <p className="text-red-500 text-sm mt-2">{error}</p>}
                    <div className="flex justify-end gap-3 mt-6">
                        <button type="button" onClick={onClose} className="px-4 py-2 rounded-md text-sm font-medium hover:bg-gray-100 dark:hover:bg-gray-700">Cancel</button>
                        <button type="submit" className="px-4 py-2 rounded-md text-sm font-medium bg-blue-600 text-white hover:bg-blue-700 disabled:bg-blue-400" disabled={!newName.trim()}>Rename</button>
                    </div>
                </form>
            </div>
        </div>
    );
};
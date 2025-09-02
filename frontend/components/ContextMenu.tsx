'use client';

import { Pencil, Trash2 } from 'lucide-react';

interface ContextMenuProps {
    x: number;
    y: number;
    onRename: () => void;
    onDelete: () => void;
}

export const ContextMenu = ({ x, y, onRename, onDelete }: ContextMenuProps) => {
    return (
        <div 
            className="absolute bg-white dark:bg-gray-800 rounded-md shadow-lg z-20 border dark:border-gray-700 py-1"
            style={{ top: y, left: x }}
        >
            <button onClick={onRename} className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-3"><Pencil className="h-4 w-4" /><span>Rename</span></button>
            <button onClick={onDelete} className="w-full text-left px-4 py-2 text-sm text-red-600 dark:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 flex items-center gap-3"><Trash2 className="h-4 w-4" /><span>Delete</span></button>
        </div>
    );
};
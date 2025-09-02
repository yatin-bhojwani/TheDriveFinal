'use client';

import { useRef, useState, useEffect } from 'react';
import { Search, UploadCloud, FolderPlus, Plus, MessageSquare, LogOut } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';

interface HeaderProps {
    onUpload: (file: File) => void;
    onCreateFolder: () => void;
    onToggleChat: () => void;
}

export const Header = ({ onUpload, onCreateFolder, onToggleChat }: HeaderProps) => {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [isDropdownOpen, setDropdownOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);
    const router = useRouter();

    const handleSignOut = async () => {
        try {
            await api.logout(); // Call the backend to clear the cookie
            router.push('/login');
        } catch (error) {
            console.error("Logout failed:", error);
        }
    };
    
    const handleUploadClick = () => {
        fileInputRef.current?.click();
        setDropdownOpen(false);
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            onUpload(file);
        }
    };
    
    const handleCreateFolderClick = () => {
        onCreateFolder();
        setDropdownOpen(false);
    };

    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setDropdownOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, [dropdownRef]);

    return (
        <header className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50 backdrop-blur-sm">
            <div className="flex items-center gap-4">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-500 h-8 w-8"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path></svg>
                <h1 className="text-xl font-bold">TheDrive</h1>
            </div>
            <div className="flex-1 max-w-xl">
                <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
                    <input type="search" placeholder="Search files, folders, or ask anything..." className="w-full pl-10 pr-4 py-2 rounded-full bg-gray-100 dark:bg-gray-700 border border-transparent focus:bg-white dark:focus:bg-gray-800 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/50 outline-none transition-all" />
                </div>
            </div>
            <div className="flex items-center gap-4">
                <div className="relative" ref={dropdownRef}>
                    <button 
                        onClick={() => setDropdownOpen(!isDropdownOpen)}
                        className="p-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition-colors shadow-md hover:shadow-lg">
                        <Plus className="h-6 w-6" />
                    </button>
                    {isDropdownOpen && (
                        <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-md shadow-lg z-10 border dark:border-gray-700 py-1">
                            <button onClick={handleUploadClick} className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-3"><UploadCloud className="h-4 w-4" /><span>Upload File</span></button>
                            <button onClick={handleCreateFolderClick} className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-3"><FolderPlus className="h-4 w-4" /><span>Create Folder</span></button>
                        </div>
                    )}
                </div>
                <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" />
                <button onClick={onToggleChat} className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700"><MessageSquare className="h-5 w-5" /></button>
                <div className="w-10 h-10 rounded-full bg-gray-300 dark:bg-gray-600 flex items-center justify-center font-bold text-lg">A</div>
                <button onClick={handleSignOut} className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700" title="Sign Out">
                    <LogOut className="h-5 w-5 text-red-500" />
                </button>
            </div>
        </header>
    );
};
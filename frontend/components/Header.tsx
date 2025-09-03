'use client';

import React from 'react';
import { MessageSquare, LogOut } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';

interface HeaderProps {
    onToggleChat: () => void;
}

export const Header = ({ onToggleChat }: HeaderProps) => {
    const router = useRouter();

    const handleSignOut = async () => {
        try {
            await api.logout(); // Call the backend to clear the cookie
            router.push('/login');
        } catch (error) {
            console.error("Logout failed:", error);
        }
    };

    return (
        <header className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50 backdrop-blur-sm">
            <div className="flex items-center gap-4">
                <img 
                    src="/logo with black b.png" 
                    alt="TheDrive Logo" 
                    className="h-26 w-26 object-contain"
                    onError={(e) => {
                        // Fallback to a colored div if image doesn't load
                        const target = e.target as HTMLImageElement;
                        target.style.display = 'none';
                        const fallback = document.createElement('div');
                        fallback.className = 'h-26 w-26 bg-blue-500 rounded flex items-center justify-center text-white font-bold text-lg';
                        fallback.textContent = 'TD';
                        target.parentNode?.insertBefore(fallback, target);
                    }}
                />
                <h1 className="text-xl font-bold">TheDrive</h1>
            </div>
            <div className="flex items-center gap-4">
                <button onClick={onToggleChat} className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700"><MessageSquare className="h-5 w-5" /></button>
                <div className="w-10 h-10 rounded-full bg-gray-300 dark:bg-gray-600 flex items-center justify-center font-bold text-lg">A</div>
                <button onClick={handleSignOut} className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700" title="Sign Out">
                    <LogOut className="h-5 w-5 text-red-500" />
                </button>
            </div>
        </header>
    );
};
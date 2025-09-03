'use client';

import { useRef, useState, useEffect } from 'react';
import { UploadCloud, FolderPlus, Plus, X } from 'lucide-react';

interface FloatingActionButtonProps {
  onUpload: (file: File) => void;
  onCreateFolder: () => void;
}

export const FloatingActionButton = ({ onUpload, onCreateFolder }: FloatingActionButtonProps) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const fabRef = useRef<HTMLDivElement>(null);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
    setIsExpanded(false);
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onUpload(file);
    }
  };

  const handleCreateFolderClick = () => {
    onCreateFolder();
    setIsExpanded(false);
  };

  // Close when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (fabRef.current && !fabRef.current.contains(event.target as Node)) {
        setIsExpanded(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  return (
    <div 
      ref={fabRef}
      className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-3"
    >
      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={handleFileChange} 
        className="hidden" 
      />
      
      {/* Action buttons - shown when expanded */}
      <div className={`flex flex-col gap-3 transition-all duration-200 ${
        isExpanded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2 pointer-events-none'
      }`}>
        <button
          onClick={handleUploadClick}
          className="flex items-center gap-3 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-200 px-4 py-3 rounded-full shadow-lg border border-gray-200 dark:border-gray-600 transition-all duration-200 hover:scale-105"
        >
          <UploadCloud className="h-5 w-5" />
          <span className="font-medium">Upload File</span>
        </button>
        
        <button
          onClick={handleCreateFolderClick}
          className="flex items-center gap-3 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-200 px-4 py-3 rounded-full shadow-lg border border-gray-200 dark:border-gray-600 transition-all duration-200 hover:scale-105"
        >
          <FolderPlus className="h-5 w-5" />
          <span className="font-medium">Create Folder</span>
        </button>
      </div>
      
      {/* Main FAB button */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={`flex items-center justify-center w-14 h-14 bg-blue-600 hover:bg-blue-700 text-white rounded-full shadow-lg transition-all duration-200 hover:scale-110 ${
          isExpanded ? 'rotate-45' : 'rotate-0'
        }`}
      >
        {isExpanded ? <X className="h-6 w-6" /> : <Plus className="h-6 w-6" />}
      </button>
    </div>
  );
};
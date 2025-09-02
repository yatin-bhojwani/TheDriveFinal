'use client';

import { Folder } from 'lucide-react';
import { FileIcon } from './FileIcon';
import type { FileSystemItem } from '@/types';

interface FileItemProps {
  item: FileSystemItem;
  isSelected: boolean;
  onItemClick: (item: FileSystemItem) => void;
  onItemDoubleClick: (item: FileSystemItem) => void;
  onContextMenu: (event: React.MouseEvent, item: FileSystemItem) => void;
}

export const FileGridItem = ({ item, isSelected, onItemClick, onItemDoubleClick, onContextMenu }: FileItemProps) => {
  return (
    <div 
      className={`group relative bg-white dark:bg-gray-800 rounded-lg shadow-sm hover:shadow-lg transition-all duration-300 cursor-pointer p-4 flex flex-col justify-between border ${isSelected ? 'border-blue-500 ring-2 ring-blue-500/50' : 'border-gray-200 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-500'}`}
      onClick={() => onItemClick(item)}
      onDoubleClick={() => onItemDoubleClick(item)}
      onContextMenu={(e) => onContextMenu(e, item)}
    >
      <div className="flex items-center gap-3 mb-4">
        {item.type === 'folder' ? (
          <Folder className="w-6 h-6 text-blue-400" />
        ) : (
          <FileIcon fileType={item.fileType} />
        )}
        <span className="font-medium truncate flex-1">{item.name}</span>
      </div>
      <div className="text-xs text-gray-500 dark:text-gray-400">
        <p>Last modified: {item.lastModified}</p>
        {item.size && <p>Size: {item.size}</p>}
      </div>
    </div>
  );
};
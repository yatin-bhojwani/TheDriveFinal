'use client';

import { Folder } from 'lucide-react';
import { FileIcon } from './FileIcon';
import type { FileSystemItem } from '@/types';

interface FileListItemProps {
  item: FileSystemItem;
  isSelected: boolean;
  onItemClick: (item: FileSystemItem) => void;
  onItemDoubleClick: (item: FileSystemItem) => void;
  onContextMenu: (event: React.MouseEvent, item: FileSystemItem) => void;
}

export const FileListItem = ({ item, isSelected, onItemClick, onItemDoubleClick, onContextMenu }: FileListItemProps) => {
    return (
        <div 
            className={`group flex items-center w-full p-2 rounded-lg cursor-pointer ${isSelected ? 'bg-blue-100 dark:bg-blue-900/50' : 'hover:bg-gray-100 dark:hover:bg-gray-800'}`}
            onClick={() => onItemClick(item)}
            onDoubleClick={() => onItemDoubleClick(item)}
            onContextMenu={(e) => onContextMenu(e, item)}
        >
            <div className="flex items-center gap-3 w-1/2">
                {item.type === 'folder' ? (
                    <Folder className="w-6 h-6 text-blue-400 flex-shrink-0" />
                ) : (
                    <FileIcon fileType={item.fileType} />
                )}
                <span className="font-medium truncate">{item.name}</span>
            </div>
            <div className="w-1/4 text-sm text-gray-500 dark:text-gray-400">{item.lastModified}</div>
            <div className="w-1/4 text-sm text-gray-500 dark:text-gray-400">{item.size || '--'}</div>
        </div>
    );
};
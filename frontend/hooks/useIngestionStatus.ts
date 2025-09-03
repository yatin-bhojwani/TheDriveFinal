import { useEffect, useRef } from 'react';
import { api } from '@/lib/api';
import type { FileSystemItem } from '@/types';

interface UseIngestionStatusProps {
  items: FileSystemItem[];
  onItemsUpdate: (items: FileSystemItem[]) => void;
}

export function useIngestionStatus({ items, onItemsUpdate }: UseIngestionStatusProps) {
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    // Find files that are currently being processed or pending
    const processingFiles = items.filter(
      item => 
        item.type === 'file' && 
        (item.ingestion_status === 'processing' || item.ingestion_status === 'pending')
    );

    // If there are processing files, start polling
    if (processingFiles.length > 0) {
      intervalRef.current = setInterval(async () => {
        try {
          // Check status for each processing file
          const statusChecks = processingFiles.map(async (file) => {
            try {
              const status = await api.getIngestionStatus(file.id);
              return { fileId: file.id, status: status.ingestion_status as FileSystemItem['ingestion_status'] };
            } catch (error) {
              console.warn(`Failed to check status for file ${file.id}:`, error);
              return null;
            }
          });

          const results = await Promise.all(statusChecks);
          
          // Update items with new statuses
          let hasUpdates = false;
          const updatedItems = items.map(item => {
            const result = results.find(r => r?.fileId === item.id);
            if (result && result.status !== item.ingestion_status) {
              hasUpdates = true;
              return { ...item, ingestion_status: result.status };
            }
            return item;
          });

          if (hasUpdates) {
            onItemsUpdate(updatedItems);
          }

          // Stop polling if no files are processing anymore
          const stillProcessing = updatedItems.some(
            item => 
              item.type === 'file' && 
              (item.ingestion_status === 'processing' || item.ingestion_status === 'pending')
          );

          if (!stillProcessing) {
            if (intervalRef.current) {
              clearInterval(intervalRef.current);
              intervalRef.current = null;
            }
          }
        } catch (error) {
          console.error('Error checking ingestion status:', error);
        }
      }, 3000); // Check every 3 seconds
    } else {
      // No processing files, clear any existing interval
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    // Cleanup on unmount or dependencies change
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [items, onItemsUpdate]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);
}

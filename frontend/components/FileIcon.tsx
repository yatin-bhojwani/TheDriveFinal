import { FileText, FileCode, FileImage, FileSpreadsheet } from 'lucide-react';
import type { FileSystemItem } from '@/types';

export const FileIcon = ({ fileType }: { fileType?: FileSystemItem['fileType'] }) => {
  switch (fileType) {
    case 'pdf': return <FileText className="w-6 h-6 text-red-500" />;
    case 'docx': return <FileText className="w-6 h-6 text-blue-500" />;
    case 'xlsx': return <FileSpreadsheet className="w-6 h-6 text-green-500" />;
    case 'png': return <FileImage className="w-6 h-6 text-purple-500" />;
    case 'js': return <FileCode className="w-6 h-6 text-yellow-500" />;
    default: return <FileText className="w-6 h-6 text-gray-500" />;
  }
};
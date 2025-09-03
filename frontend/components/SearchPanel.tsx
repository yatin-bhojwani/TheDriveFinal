'use client';

import { useState, useEffect } from 'react';
import { Search, Filter, X, Calendar, FileType, HardDrive } from 'lucide-react';
import type { FileSystemItem, SearchFilters } from '@/types';
import { api } from '@/lib/api';

interface SearchPanelProps {
  onSearchResults: (results: FileSystemItem[]) => void;
  onClearSearch: () => void;
  isSearchMode: boolean;
}

export const SearchPanel = ({ onSearchResults, onClearSearch, isSearchMode }: SearchPanelProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState<SearchFilters>({});
  const [isSearching, setIsSearching] = useState(false);

  const handleSearch = async () => {
    if (!searchQuery.trim() && Object.keys(filters).length === 0) {
      onClearSearch();
      return;
    }

    setIsSearching(true);
    try {
      const searchFilters: SearchFilters = {
        ...filters,
        query: searchQuery.trim() || undefined,
      };

      const results = await api.searchItems(searchFilters);
      onSearchResults(results);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setIsSearching(false);
    }
  };

  const handleClearAll = () => {
    setSearchQuery('');
    setFilters({});
    onClearSearch();
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const updateFilter = (key: keyof SearchFilters, value: any) => {
    setFilters(prev => ({
      ...prev,
      [key]: value || undefined
    }));
  };

  const getFilterCount = () => {
    return Object.keys(filters).filter(key => 
      filters[key as keyof SearchFilters] !== undefined && 
      filters[key as keyof SearchFilters] !== ''
    ).length;
  };

  return (
    <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
      {/* Search Bar */}
      <div className="p-4">
        <div className="flex items-center gap-2">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5" />
            <input
              type="text"
              placeholder="Search files and folders..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className={`relative p-2 rounded-lg border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700 ${
              getFilterCount() > 0 ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-300 dark:border-blue-600' : ''
            }`}
            title="Advanced filters"
          >
            <Filter className="h-5 w-5" />
            {getFilterCount() > 0 && (
              <span className="absolute -top-1 -right-1 bg-blue-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
                {getFilterCount()}
              </span>
            )}
          </button>
          
          <button
            onClick={handleSearch}
            disabled={isSearching}
            className="px-4 py-2 bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white rounded-lg transition-colors"
          >
            {isSearching ? 'Searching...' : 'Search'}
          </button>
          
          {isSearchMode && (
            <button
              onClick={handleClearAll}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              title="Clear search"
            >
              <X className="h-5 w-5" />
            </button>
          )}
        </div>
      </div>

      {/* Advanced Filters */}
      {isExpanded && (
        <div className="px-4 pb-4 border-t border-gray-200 dark:border-gray-700">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-4">
            
            {/* File Type Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                <FileType className="inline h-4 w-4 mr-1" />
                File Type
              </label>
              <select
                value={filters.file_type || ''}
                onChange={(e) => updateFilter('file_type', e.target.value)}
                className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              >
                <option value="">All types</option>
                <option value="document">Documents</option>
                <option value="image">Images</option>
                <option value="video">Videos</option>
                <option value="audio">Audio</option>
                <option value="pdf">PDF</option>
                <option value="docx">Word Documents</option>
                <option value="xlsx">Excel Files</option>
                <option value="txt">Text Files</option>
              </select>
            </div>

            {/* Item Type Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                <HardDrive className="inline h-4 w-4 mr-1" />
                Item Type
              </label>
              <select
                value={filters.item_type || ''}
                onChange={(e) => updateFilter('item_type', e.target.value)}
                className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              >
                <option value="">Files & Folders</option>
                <option value="file">Files Only</option>
                <option value="folder">Folders Only</option>
              </select>
            </div>

            {/* Ingestion Status Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Processing Status
              </label>
              <select
                value={filters.ingestion_status || ''}
                onChange={(e) => updateFilter('ingestion_status', e.target.value)}
                className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              >
                <option value="">All statuses</option>
                <option value="completed">Ready for AI</option>
                <option value="processing">Processing</option>
                <option value="pending">Pending</option>
                <option value="failed">Failed</option>
              </select>
            </div>

            {/* Date Range */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                <Calendar className="inline h-4 w-4 mr-1" />
                Upload Date From
              </label>
              <input
                type="date"
                value={filters.date_from || ''}
                onChange={(e) => updateFilter('date_from', e.target.value ? new Date(e.target.value).toISOString() : '')}
                className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Upload Date To
              </label>
              <input
                type="date"
                value={filters.date_to || ''}
                onChange={(e) => updateFilter('date_to', e.target.value ? new Date(e.target.value).toISOString() : '')}
                className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              />
            </div>

            {/* File Size Range */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Min Size (MB)
              </label>
              <input
                type="number"
                min="0"
                step="0.1"
                value={filters.min_size ? (filters.min_size / 1024 / 1024).toFixed(1) : ''}
                onChange={(e) => updateFilter('min_size', e.target.value ? parseFloat(e.target.value) * 1024 * 1024 : undefined)}
                className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                placeholder="0"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Max Size (MB)
              </label>
              <input
                type="number"
                min="0"
                step="0.1"
                value={filters.max_size ? (filters.max_size / 1024 / 1024).toFixed(1) : ''}
                onChange={(e) => updateFilter('max_size', e.target.value ? parseFloat(e.target.value) * 1024 * 1024 : undefined)}
                className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                placeholder="∞"
              />
            </div>

            {/* Clear Filters Button */}
            <div className="flex items-end">
              <button
                onClick={() => setFilters({})}
                className="w-full p-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                Clear Filters
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
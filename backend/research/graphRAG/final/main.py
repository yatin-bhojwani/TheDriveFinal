import asyncio
import os
from pathlib import Path
from typing import Dict, Any, List
import sys
from pdf_chatter import PDFChatter, PDFChatterConfig

try:
    from multiformat_chatter import MultiFormatChatter, HighlightProcessor, CrossReferenceEngine, create_multiformat_chatter
    from drive_integration import DriveAISystem
    FULL_FEATURES = True
except ImportError:
    FULL_FEATURES = False
    print("Some advanced features not available. Using basic PDF chat mode.")

class CompleteDriveSystem:
    """Main class integrating all AI drive functionalities"""
    
    def __init__(self):
        self.drive_system = None
        self.basic_chatter = None
        self.is_initialized = False
        self.use_basic_mode = not FULL_FEATURES
        
    async def initialize_system(self, neo4j_config: Dict[str, str], google_api_key: str):
        """Initialize the complete drive system"""
        try:
            if self.use_basic_mode:
                config = PDFChatterConfig(
                    neo4j_uri=neo4j_config.get('uri', 'bolt://localhost:7687'),
                    neo4j_user=neo4j_config['user'],
                    neo4j_password=neo4j_config['password'],
                    google_api_key=google_api_key,
                    gemini_model="gemini-2.5-pro"
                )
                self.basic_chatter = PDFChatter(config)
                await self.basic_chatter.initialize()
            else:
                self.drive_system = DriveAISystem(neo4j_config, google_api_key)
                await self.drive_system.initialize()
            
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Initialization failed: {e}")
            return False
    
    async def process_folder(self, folder_path: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """Process a folder with intelligent tracking"""
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        if self.use_basic_mode:
            return await self.basic_chatter.process_folder(folder_path, force_reprocess)
        else:
            return await self.drive_system.add_folder_to_drive(folder_path)
    
    async def chat_with_folder(self, folder_path: str, question: str) -> Dict[str, Any]:
        """Chat with folder content"""
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        if self.use_basic_mode:
            return await self.basic_chatter.ask_question(question)
        else:
            return await self.drive_system.chat_with_folder(folder_path, question)
    
    async def chat_with_drive(self, question: str) -> Dict[str, Any]:
        """Chat with entire drive"""
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        if self.use_basic_mode:
            return await self.basic_chatter.ask_question(question)
        else:
            return await self.drive_system.chat_with_whole_drive(question)
    
    async def analyze_highlight(self, text: str, file_path: str, context: str = "") -> Dict[str, Any]:
        """Analyze highlighted text"""
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        if self.use_basic_mode:
            return {
                'highlighted_text': text,
                'explanation': {'explanation': f"Text analysis for: {text}"},
                'file_type': 'text',
                'success': True
            }
        else:
            return await self.drive_system.analyze_highlight(text, file_path, context)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        if self.use_basic_mode:
            return {
                'mode': 'basic',
                'features': ['pdf_chat', 'intelligent_tracking'],
                'total_folders': 1 if self.basic_chatter.folder_id else 0
            }
        else:
            return await self.drive_system.get_drive_statistics()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.basic_chatter:
            await self.basic_chatter.cleanup()
        if self.drive_system:
            await self.drive_system.cleanup()

async def interactive_session(system: CompleteDriveSystem):
    """Interactive chat session"""
    print("\n🚀 AI Drive System Ready!")
    print("Commands:")
    print("  /folder <path> <question> - Chat with folder")
    print("  /drive <question>         - Chat with entire drive")
    print("  /highlight <text> <file>  - Analyze highlighted text")
    print("  /stats                    - Show statistics")
    print("  /quit                     - Exit")
    
    while True:
        try:
            user_input = input("\n💬 > ").strip()
            
            if not user_input or user_input.lower() in ['/quit', 'quit', 'exit']:
                break
            
            parts = user_input.split(' ', 2)
            command = parts[0].lower()
            
            if command == '/folder' and len(parts) >= 3:
                folder_path = parts[1]
                question = parts[2]
                result = await system.chat_with_folder(folder_path, question)
                
                print(f"\n📝 Question: {question}")
                print(f"📁 Folder: {Path(folder_path).name}")
                print(f"🤖 Answer: {result.get('answer', 'No answer available')}")
                
            elif command == '/drive' and len(parts) >= 2:
                question = ' '.join(parts[1:])
                result = await system.chat_with_drive(question)
                
                print(f"\n📝 Question: {question}")
                print(f"🤖 Answer: {result.get('answer', 'No answer available')}")
                
            elif command == '/highlight' and len(parts) >= 3:
                text = parts[1]
                file_path = parts[2] if len(parts) > 2 else "unknown"
                result = await system.analyze_highlight(text, file_path)
                
                print(f"\n🖍️ Highlighted: {text}")
                print(f"🤖 Explanation: {result.get('explanation', {}).get('explanation', 'No explanation available')}")
                
            elif command == '/stats':
                stats = await system.get_statistics()
                print(f"\n📊 System Statistics:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                
            else:
                print("Invalid command. Use /folder, /drive, /highlight, /stats, or /quit")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n👋 Goodbye!")

async def main():
    """Main function"""
    print("🚀 AI-Powered Drive System")
    print("=" * 50)
    
    # Configuration
    neo4j_config = {
        'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        'user': os.getenv('NEO4J_USER', 'neo4j'),
        'password': os.getenv('NEO4J_PASSWORD', 'password')
    }
    google_api_key = os.getenv('GOOGLE_API_KEY')
    
    if not google_api_key:
        print("❌ Please set GOOGLE_API_KEY environment variable")
        return
    
    # Initialize system
    system = CompleteDriveSystem()
    print(f"📦 Mode: {'Basic PDF Chat' if system.use_basic_mode else 'Full Multi-Format'}")
    
    success = await system.initialize_system(neo4j_config, google_api_key)
    if not success:
        print("❌ Failed to initialize system")
        return
    
    print("✅ System initialized successfully!")
    
    # Process folders from command line
    folder_paths = sys.argv[1:] if len(sys.argv) > 1 else []
    
    if folder_paths:
        print(f"\n📁 Processing {len(folder_paths)} folders...")
        
        for folder_path in folder_paths:
            if Path(folder_path).exists():
                print(f"Processing: {folder_path}")
                result = await system.process_folder(folder_path)
                
                if result.get('status') in ['success', 'already_processed']:
                    print(f"✅ {Path(folder_path).name}: {result.get('documents_processed', 0)} documents")
                else:
                    print(f"❌ {Path(folder_path).name}: {result.get('error', 'Unknown error')}")
            else:
                print(f"❌ Folder not found: {folder_path}")
        
        # Show statistics
        stats = await system.get_statistics()
        print(f"\n📊 Total entities: {stats.get('total_entities', 'N/A')}")
        print(f"📊 Total relationships: {stats.get('total_relationships', 'N/A')}")
        
        # Start interactive session
        await interactive_session(system)
    
    else:
        print("\n💡 Usage:")
        print("   python main.py <folder1> <folder2> ...")
        print("   Example: python main.py ./documents ./data")
        print("\n🔧 Environment variables:")
        print("   NEO4J_USER, NEO4J_PASSWORD, GOOGLE_API_KEY")
        print("   Optional: NEO4J_URI")
        
        # Demo with manual folder input
        print("\n🔄 Enter a folder path to process (or 'quit'):")
        while True:
            folder_input = input("📁 Folder path: ").strip()
            if folder_input.lower() in ['quit', 'exit', '']:
                break
            
            if Path(folder_input).exists():
                result = await system.process_folder(folder_input)
                if result.get('status') in ['success', 'already_processed']:
                    print(f"✅ Processed successfully!")
                    await interactive_session(system)
                    break
                else:
                    print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            else:
                print(f"❌ Folder not found: {folder_input}")
    
    # Cleanup
    await system.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

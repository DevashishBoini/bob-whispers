"""
Comprehensive Test for Semantic Memory Integration

This test verifies that:
1. Messages are stored in semantic memory
2. Semantic search retrieves relevant past conversations
3. Retrieved context is actually used in AI responses
4. The system can reference early messages when answering new queries

Test Scenario:
- Create 35-40 chat messages across different topics
- Include specific technical details in early messages
- Ask a new question that should reference early conversations
- Verify the AI response includes information from those early messages
"""

import time
import json
import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import os
import sys

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from src.services.enhanced_conversation_service import (
    EnhancedConversationService,
    init_enhanced_conversation_service
)
from src.services.memory_service import init_memory_service
from src.services.semantic_search import init_search_engine

class SemanticMemoryTester:
    """Test class for comprehensive semantic memory verification."""
    
    def __init__(self):
        """Initialize the semantic memory integration tester."""
        self.service = None
        self.conversation_id = ""
        self.rate_limit_delay = 2.0  # 2 seconds between API calls
        self.batch_delay = 5.0  # 5 seconds between batches
        self.test_results = {
            "test_start_time": datetime.now().isoformat(),
            "test_end_time": None,
            "conversation_id": None,
            "messages_created": [],
            "semantic_searches_performed": 0,
            "rate_limit_info": {
                "delays_applied": [],
                "total_delay_time": 0.0,
                "api_calls_made": 0
            },
            "semantic_memory_tests": [],
            "context_usage_tests": [],
            "ai_response_analysis": [],
            "test_success": False,
            "error_message": None
        }
        
    def create_test_conversation_history(self) -> List[Dict[str, str]]:
        """
        Create a comprehensive conversation history with specific technical details
        that can be referenced later.
        
        Returns:
            List[Dict]: Test conversation messages
        """
        
        # Early messages with specific technical details (messages 1-10)
        early_technical_messages = [
            {
                "user": "How do I set up FastAPI with proper error handling?",
                "assistant": "To set up FastAPI with proper error handling, you should create custom exception handlers using @app.exception_handler decorator. For example, you can handle ValidationError exceptions and return structured JSON responses with proper HTTP status codes like 422 for validation errors."
            },
            {
                "user": "What's the best way to handle database connections in FastAPI?",
                "assistant": "Use dependency injection with SQLAlchemy Session. Create a get_db() dependency that yields a database session, and use Depends(get_db) in your route parameters. Always use try-finally blocks or context managers to ensure connections are properly closed. Consider using connection pooling for production."
            },
            {
                "user": "How do I implement JWT authentication in FastAPI?",
                "assistant": "For JWT authentication in FastAPI: 1) Install python-jose[cryptography] and passlib[bcrypt], 2) Create JWT tokens using jose.jwt.encode() with a SECRET_KEY, 3) Use OAuth2PasswordBearer for token extraction, 4) Create a dependency to verify tokens and extract user info, 5) Protect routes with Depends(get_current_user)."
            },
            {
                "user": "What are the best practices for FastAPI project structure?",
                "assistant": "Best FastAPI project structure: app/ (main application), models/ (SQLAlchemy models), schemas/ (Pydantic models), crud/ (database operations), api/ (route handlers), core/ (configuration, security), tests/ (test files). Use dependency injection, separate concerns, and follow domain-driven design patterns."
            },
            {
                "user": "How do I optimize FastAPI performance for production?",
                "assistant": "FastAPI performance optimization: 1) Use async/await for I/O operations, 2) Enable Gzip compression, 3) Implement proper database connection pooling, 4) Use Redis for caching, 5) Add request/response middleware for monitoring, 6) Configure Uvicorn with multiple workers, 7) Use CDN for static assets, 8) Implement proper logging and monitoring."
            },
            {
                "user": "What's the difference between Pydantic v1 and v2?",
                "assistant": "Pydantic v2 major changes: 1) Core rewritten in Rust for 5-50x performance improvement, 2) New BaseModel configuration using model_config instead of Config class, 3) Field validation changes - validator decorator replaced with field_validator, 4) Serialization improvements with new serialization context, 5) Better error messages and debugging, 6) Breaking changes in JSON schema generation."
            },
            {
                "user": "How do I handle file uploads in FastAPI?",
                "assistant": "FastAPI file uploads: Use UploadFile parameter with File() dependency. For single files: async def upload(file: UploadFile = File(...)). For multiple files: files: List[UploadFile] = File(...). Read content with await file.read() or file.file.read(). Validate file types, implement size limits, and store files securely. Use Form() for additional form data with file uploads."
            },
            {
                "user": "What's the best way to test FastAPI applications?",
                "assistant": "FastAPI testing best practices: 1) Use pytest with TestClient from fastapi.testclient, 2) Create test database with SQLAlchemy fixtures, 3) Use dependency overrides for mocking, 4) Test both sync and async endpoints, 5) Implement integration tests for database operations, 6) Use parametrized tests for multiple scenarios, 7) Mock external API calls, 8) Test authentication and authorization flows."
            },
            {
                "user": "How do I implement proper logging in FastAPI?",
                "assistant": "FastAPI logging setup: 1) Configure Python logging module with proper formatters, 2) Use structured logging with JSON format for production, 3) Implement request/response logging middleware, 4) Log different levels (DEBUG, INFO, WARNING, ERROR), 5) Use correlation IDs for request tracing, 6) Integrate with monitoring tools like ELK stack or Prometheus, 7) Avoid logging sensitive information."
            },
            {
                "user": "What are FastAPI background tasks and when to use them?",
                "assistant": "FastAPI BackgroundTasks allow running functions after returning response. Use for: email sending, file processing, cleanup operations. Example: def send_email(email: str): pass; async def endpoint(background_tasks: BackgroundTasks): background_tasks.add_task(send_email, 'user@example.com'). For heavy tasks, consider Celery or RQ instead."
            }
        ]
        
        # Middle messages on different topics (messages 11-25)
        middle_diverse_messages = [
            {
                "user": "How do I learn Python effectively?",
                "assistant": "Learn Python effectively: 1) Start with basics (variables, functions, data structures), 2) Practice with coding challenges on platforms like LeetCode or HackerRank, 3) Build real projects, 4) Read Python documentation and PEPs, 5) Join Python communities, 6) Learn frameworks like Django or Flask, 7) Understand Python's philosophy and idioms."
            },
            {
                "user": "What's the difference between lists and tuples?",
                "assistant": "Lists vs Tuples: Lists are mutable (can be modified), use square brackets [], support append/remove operations. Tuples are immutable (cannot be changed), use parentheses (), are hashable and can be dictionary keys. Tuples are faster for iteration and use less memory."
            },
            {
                "user": "How do decorators work in Python?",
                "assistant": "Python decorators are functions that modify other functions. They use @decorator syntax. A decorator takes a function as argument and returns a modified function. Common use cases: logging, timing, authentication, caching. Example: @functools.wraps preserves original function metadata."
            },
            {
                "user": "What are Python context managers?",
                "assistant": "Context managers handle resource management using 'with' statement. They implement __enter__ and __exit__ methods. Common example: 'with open(file) as f:' automatically closes file. You can create custom context managers using @contextmanager decorator or by implementing the context manager protocol."
            },
            {
                "user": "How do I handle exceptions in Python?",
                "assistant": "Python exception handling: Use try-except blocks, catch specific exceptions rather than bare except, use finally for cleanup, raise exceptions with descriptive messages. Example: try: risky_operation() except ValueError as e: handle_error(e) finally: cleanup(). Use custom exceptions for application-specific errors."
            },
            {
                "user": "What's the difference between == and is in Python?",
                "assistant": "== checks value equality (calls __eq__ method), 'is' checks identity (same object in memory). Use == for comparing values, 'is' for comparing with None, True, False, or checking if two variables reference the same object. Example: [1,2,3] == [1,2,3] is True, but [1,2,3] is [1,2,3] is False."
            },
            {
                "user": "How do I use virtual environments?",
                "assistant": "Python virtual environments isolate project dependencies. Create with: python -m venv myenv, activate with: source myenv/bin/activate (Linux/Mac) or myenv\\Scripts\\activate (Windows). Install packages with pip, save dependencies with pip freeze > requirements.txt. Deactivate with 'deactivate' command."
            },
            {
                "user": "What are Python generators?",
                "assistant": "Generators are functions that yield values lazily, saving memory. They use 'yield' keyword instead of 'return'. Generator expressions: (x*2 for x in range(10)). They're iterables that produce values on-demand. Useful for processing large datasets or infinite sequences. Once exhausted, they can't be reused."
            },
            {
                "user": "How do I work with JSON in Python?",
                "assistant": "Python JSON handling: import json module. json.loads() converts JSON string to Python object, json.dumps() converts Python object to JSON string. json.load() reads from file, json.dump() writes to file. Handle encoding issues with ensure_ascii=False. Use custom encoders for complex objects."
            },
            {
                "user": "What are lambda functions?",
                "assistant": "Lambda functions are anonymous functions defined with 'lambda' keyword. Syntax: lambda arguments: expression. Example: square = lambda x: x**2. Commonly used with map(), filter(), sort(). Limited to single expressions. Use regular functions for complex logic."
            },
            {
                "user": "How do I debug Python code?",
                "assistant": "Python debugging methods: 1) Use print statements for simple debugging, 2) Python debugger (pdb): import pdb; pdb.set_trace(), 3) IDE debuggers with breakpoints, 4) Logging for production debugging, 5) Unit tests to isolate issues, 6) Static analysis tools like pylint, 7) Profiling tools for performance issues."
            },
            {
                "user": "What are Python metaclasses?",
                "assistant": "Metaclasses are classes whose instances are classes. They control class creation. 'type' is the default metaclass. Use __new__ and __init__ methods in metaclasses. Common use: ORMs, singletons, attribute validation. Generally avoid unless you have a specific need - 'if you're not sure you need metaclasses, you don't'."
            },
            {
                "user": "How do I handle dates and times?",
                "assistant": "Python datetime handling: Use datetime module. datetime.now() for current time, strptime() to parse strings, strftime() to format. Use timezone-aware datetime with pytz or zoneinfo. For date arithmetic, use timedelta. Consider using arrow or pendulum libraries for advanced date operations."
            },
            {
                "user": "What are Python type hints?",
                "assistant": "Type hints specify expected types for variables, function parameters, and return values. Use typing module for complex types. Example: def greet(name: str) -> str: return f'Hello {name}'. Benefits: better IDE support, static type checking with mypy, self-documenting code. Optional in Python but recommended for larger projects."
            },
            {
                "user": "How do I optimize Python performance?",
                "assistant": "Python performance optimization: 1) Profile first with cProfile, 2) Use built-in functions and libraries, 3) List comprehensions over loops, 4) Avoid global variables, 5) Use generators for large datasets, 6) Consider NumPy for numerical operations, 7) Cython or numba for CPU-intensive tasks, 8) Async programming for I/O bound tasks."
            }
        ]
        
        # Recent messages on various topics (messages 26-35)
        recent_messages = [
            {
                "user": "What's machine learning?",
                "assistant": "Machine learning is a subset of AI where computers learn patterns from data without explicit programming. Types: supervised (labeled data), unsupervised (pattern finding), reinforcement (reward-based). Common algorithms: linear regression, decision trees, neural networks. Applications: image recognition, NLP, recommendation systems."
            },
            {
                "user": "How do neural networks work?",
                "assistant": "Neural networks mimic brain neurons with interconnected nodes. Input layer receives data, hidden layers process it through weighted connections and activation functions, output layer produces results. Learning happens through backpropagation, adjusting weights based on errors. Deep learning uses multiple hidden layers."
            },
            {
                "user": "What's the difference between AI and ML?",
                "assistant": "AI (Artificial Intelligence) is the broad concept of machines performing tasks that require human intelligence. ML (Machine Learning) is a subset of AI focused on learning from data. AI includes rule-based systems, while ML specifically learns patterns. Deep learning is a subset of ML using neural networks."
            },
            {
                "user": "What are REST APIs?",
                "assistant": "REST APIs are web services following Representational State Transfer principles. They use HTTP methods (GET, POST, PUT, DELETE) for operations, have stateless communication, use JSON for data exchange, and follow resource-based URLs. RESTful design emphasizes scalability and simplicity."
            },
            {
                "user": "How do databases work?",
                "assistant": "Databases store and organize data for efficient retrieval. Relational databases use tables with rows/columns, connected by relationships. SQL manages relational data. NoSQL databases (document, key-value, graph) handle unstructured data. ACID properties ensure data integrity. Indexing improves query performance."
            },
            {
                "user": "What's version control?",
                "assistant": "Version control tracks changes in code over time. Git is the most popular system. Key concepts: repositories, commits, branches, merges. Enables collaboration, rollback to previous versions, and parallel development. Distributed model means each developer has complete history locally."
            },
            {
                "user": "What's cloud computing?",
                "assistant": "Cloud computing delivers computing services over the internet. Types: IaaS (infrastructure), PaaS (platform), SaaS (software). Benefits: scalability, cost-effectiveness, accessibility. Major providers: AWS, Azure, Google Cloud. Enables on-demand resource allocation and global deployment."
            },
            {
                "user": "How does encryption work?",
                "assistant": "Encryption converts readable data into unreadable format using algorithms and keys. Symmetric encryption uses same key for encryption/decryption. Asymmetric uses public/private key pairs. Hash functions create one-way fingerprints. Used for data protection, authentication, and digital signatures."
            },
            {
                "user": "What's agile development?",
                "assistant": "Agile is an iterative software development approach emphasizing collaboration, flexibility, and customer feedback. Key principles: working software over documentation, individuals over processes, customer collaboration over contracts. Common frameworks: Scrum, Kanban. Features short sprints and continuous improvement."
            },
            {
                "user": "What are design patterns?",
                "assistant": "Design patterns are reusable solutions to common programming problems. Categories: Creational (object creation), Structural (object composition), Behavioral (object interaction). Examples: Singleton, Factory, Observer, Strategy. They improve code maintainability and communication among developers."
            }
        ]
        
        # Combine all messages
        all_messages = early_technical_messages + middle_diverse_messages + recent_messages
        return all_messages
    
    async def setup_conversation_with_history(self) -> str:
        """
        Create a conversation and populate it with test history.
        
        Returns:
            str: Conversation ID
        """
        print("🚀 Setting up enhanced conversation service...")
        
        # Initialize enhanced conversation service
        self.service = init_enhanced_conversation_service()
        
        # Create new conversation
        self.conversation_id = self.service.start_new_conversation("Semantic Memory Integration Test")
        self.test_results["conversation_id"] = self.conversation_id
        
        print(f"📝 Created test conversation: {self.conversation_id}")
        
        # Get test messages
        test_messages = self.create_test_conversation_history()
        
        print(f"💬 Adding {len(test_messages)} messages to conversation history...")
        print("⏳ This will take several minutes due to rate limiting...")
        
        # Add messages to conversation with rate limiting
        for i, message_pair in enumerate(test_messages, 1):
            print(f"  Adding message {i}/{len(test_messages)}: {message_pair['user'][:50]}...")
            
            # Apply rate limiting delay
            if i > 1:  # Skip delay for first message
                delay_start = time.time()
                print(f"    ⏱️  Waiting {self.rate_limit_delay}s for rate limiting...")
                time.sleep(self.rate_limit_delay)
                actual_delay = time.time() - delay_start
                self.test_results["rate_limit_info"]["delays_applied"].append(actual_delay)
            
            try:
                # Process message through enhanced service
                response = self.service.process_message_with_memory(
                    conversation_id=self.conversation_id,
                    user_message=message_pair["user"],
                    input_type="text"
                )
                
                if response.success:
                    # Store message info for analysis
                    message_info = {
                        "sequence": i,
                        "user_message": message_pair["user"],
                        "expected_ai_response": message_pair["assistant"],
                        "actual_ai_response": response.ai_content,
                        "message_id": response.message_id,
                        "timestamp": response.timestamp.isoformat() if response.timestamp else None,
                        "token_usage": response.token_usage
                    }
                    self.test_results["messages_created"].append(message_info)
                    print(f"    ✅ Message {i} added successfully")
                else:
                    print(f"    ❌ Failed to add message {i}: {response.error_message}")
                    
            except Exception as e:
                print(f"    ❌ Error adding message {i}: {e}")
            
            # Additional delay between batches
            if i % 5 == 0:
                print(f"    📊 Completed batch {i//5}. Taking short break...")
                time.sleep(self.batch_delay)
        
        print(f"✅ Conversation history setup complete!")
        return self.conversation_id
    
    def test_semantic_search_retrieval(self) -> List[Dict[str, Any]]:
        """
        Test that semantic search can retrieve relevant past conversations.
        
        Returns:
            List[Dict]: Search test results
        """
        print("\n🔍 Testing semantic search retrieval...")
        
        search_tests = [
            {
                "query": "FastAPI error handling best practices",
                "expected_relevance": "Should find early message about @app.exception_handler decorator",
                "target_message_sequence": 1
            },
            {
                "query": "JWT authentication implementation FastAPI",
                "expected_relevance": "Should find message about python-jose and OAuth2PasswordBearer",
                "target_message_sequence": 3
            },
            {
                "query": "database connection pooling FastAPI",
                "expected_relevance": "Should find message about SQLAlchemy Session and dependency injection",
                "target_message_sequence": 2
            },
            {
                "query": "FastAPI project structure recommendations",
                "expected_relevance": "Should find message about app/, models/, schemas/ structure",
                "target_message_sequence": 4
            },
            {
                "query": "Pydantic version differences",
                "expected_relevance": "Should find message about Pydantic v1 vs v2 changes",
                "target_message_sequence": 6
            }
        ]
        
        search_results = []
        
        # Get memory service for direct testing
        memory_service = init_memory_service()
        
        for test in search_tests:
            print(f"  🎯 Testing query: '{test['query']}'")
            
            try:
                # Direct semantic search test
                relevant_memories = memory_service.get_relevant_memories(
                    conversation_id=self.conversation_id,
                    query_text=test["query"],
                    max_memories=5,
                    similarity_threshold=0.7
                )
                
                search_result = {
                    "query": test["query"],
                    "expected_relevance": test["expected_relevance"],
                    "target_sequence": test["target_message_sequence"],
                    "memories_found": len(relevant_memories),
                    "top_similarities": [],
                    "found_target": False
                }
                
                # Analyze found memories
                for memory in relevant_memories:
                    similarity_info = {
                        "message_id": memory.message_id,
                        "similarity_score": memory.similarity_score,
                        "user_message": memory.user_message[:100] + "...",
                        "timestamp": memory.timestamp.isoformat() if memory.timestamp else None
                    }
                    search_result["top_similarities"].append(similarity_info)
                    
                    # Check if we found the target message
                    if memory.message_id == test["target_message_sequence"]:
                        search_result["found_target"] = True
                
                search_results.append(search_result)
                
                print(f"    📊 Found {len(relevant_memories)} relevant memories")
                print(f"    🎯 Target found: {'✅' if search_result['found_target'] else '❌'}")
                
                if relevant_memories:
                    best_match = relevant_memories[0]
                    print(f"    🏆 Best match (score: {best_match.similarity_score:.3f}): {best_match.user_message[:50]}...")
                
            except Exception as e:
                print(f"    ❌ Search test failed: {e}")
                search_result = {
                    "query": test["query"],
                    "error": str(e)
                }
                search_results.append(search_result)
        
        self.test_results["semantic_search_tests"] = search_results
        return search_results
    
    async def test_ai_response_with_context(self) -> Dict[str, Any]:
        """
        Test that AI responses actually use the retrieved semantic context.
        
        Returns:
            Dict: Analysis of AI response context usage
        """
        print("\n🤖 Testing AI response with semantic context...")
        
        # Test query that should reference early FastAPI messages
        test_query = "Can you remind me about the specific steps for implementing JWT authentication in FastAPI that we discussed earlier? I need the exact libraries and dependencies you mentioned."
        
        print(f"  📝 Test query: '{test_query}'")
        print("  ⏳ Generating AI response (this may take a moment)...")
        
        if not self.service:
            raise RuntimeError("Service not initialized. Call setup_service() first.")
        
        # Apply rate limiting
        time.sleep(self.rate_limit_delay)
        
        try:
            # Generate response with semantic memory
            response = self.service.process_message_with_memory(
                conversation_id=self.conversation_id,
                user_message=test_query,
                input_type="text"
            )
            
            if not response.success:
                return {
                    "error": f"AI response failed: {response.error_message}",
                    "success": False
                }
            
            # Analyze response for context usage
            ai_response = response.ai_content.lower()
            
            # Check for specific references from early JWT message
            jwt_keywords = [
                "python-jose", "cryptography", "passlib", "bcrypt",
                "jose.jwt.encode", "secret_key", "oauth2passwordbearer",
                "get_current_user", "depends"
            ]
            
            found_keywords = []
            for keyword in jwt_keywords:
                if keyword.lower() in ai_response:
                    found_keywords.append(keyword)
            
            # Check for general context indicators
            context_indicators = [
                "discussed", "mentioned", "earlier", "previously", 
                "before", "as i said", "remember", "recall"
            ]
            
            found_indicators = []
            for indicator in context_indicators:
                if indicator in ai_response:
                    found_indicators.append(indicator)
            
            analysis = {
                "test_query": test_query,
                "ai_response": response.ai_content,
                "response_length": len(response.ai_content),
                "token_usage": response.token_usage,
                "timestamp": response.timestamp.isoformat() if response.timestamp else None,
                "context_analysis": {
                    "jwt_specific_keywords_found": found_keywords,
                    "jwt_keyword_count": len(found_keywords),
                    "context_indicators_found": found_indicators,
                    "context_indicator_count": len(found_indicators),
                    "likely_uses_context": len(found_keywords) >= 2 or len(found_indicators) >= 1
                },
                "success": True
            }
            
            print(f"  📊 Response length: {len(response.ai_content)} characters")
            print(f"  🔍 JWT keywords found: {len(found_keywords)}/={len(jwt_keywords)}")
            print(f"  💭 Context indicators: {found_indicators}")
            print(f"  ✅ Likely uses context: {'Yes' if analysis['context_analysis']['likely_uses_context'] else 'No'}")
            
            self.test_results["ai_response_analysis"].append(analysis)
            return analysis
            
        except Exception as e:
            print(f"  ❌ AI response test failed: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    def verify_semantic_memory_integration(self) -> Dict[str, Any]:
        """
        Verify the complete semantic memory integration.
        
        Returns:
            Dict: Verification results
        """
        print("\n✅ Verifying semantic memory integration...")
        
        verification = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": self.conversation_id,
            "total_messages_created": len(self.test_results["messages_created"]),
            "semantic_search_working": False,
            "ai_context_usage": False,
            "overall_success": False,
            "detailed_analysis": {}
        }
        
        # Check semantic search results
        search_tests = self.test_results.get("semantic_search_tests", [])
        successful_searches = sum(1 for test in search_tests if test.get("memories_found", 0) > 0)
        target_found_count = sum(1 for test in search_tests if test.get("found_target", False))
        
        verification["semantic_search_working"] = successful_searches > 0
        verification["detailed_analysis"]["search_success_rate"] = successful_searches / len(search_tests) if search_tests else 0
        verification["detailed_analysis"]["target_found_rate"] = target_found_count / len(search_tests) if search_tests else 0
        
        # Check AI response analysis
        ai_analyses = self.test_results.get("ai_response_analysis", [])
        context_usage_count = sum(1 for analysis in ai_analyses if analysis.get("context_analysis", {}).get("likely_uses_context", False))
        
        verification["ai_context_usage"] = context_usage_count > 0
        verification["detailed_analysis"]["ai_context_usage_rate"] = context_usage_count / len(ai_analyses) if ai_analyses else 0
        
        # Overall success criteria
        verification["overall_success"] = (
            verification["semantic_search_working"] and 
            verification["ai_context_usage"] and
            verification["detailed_analysis"]["search_success_rate"] > 0.6 and
            verification["detailed_analysis"]["target_found_rate"] > 0.4
        )
        
        print(f"  📊 Messages created: {verification['total_messages_created']}")
        print(f"  🔍 Semantic search working: {'✅' if verification['semantic_search_working'] else '❌'}")
        print(f"  🎯 Search success rate: {verification['detailed_analysis']['search_success_rate']:.1%}")
        print(f"  📍 Target found rate: {verification['detailed_analysis']['target_found_rate']:.1%}")
        print(f"  🤖 AI context usage: {'✅' if verification['ai_context_usage'] else '❌'}")
        print(f"  🏆 Overall success: {'✅' if verification['overall_success'] else '❌'}")
        
        self.test_results["verification_results"] = verification
        return verification
    
    def save_test_results(self, filename: str = "semantic_memory_test_results.json") -> str:
        """
        Save complete test results to JSON file.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            str: Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"semantic_memory_test_results_{timestamp}.json"
        
        # Finalize test results
        self.test_results["test_end_time"] = datetime.now().isoformat()
        total_duration = (
            datetime.fromisoformat(self.test_results["test_end_time"]) - 
            datetime.fromisoformat(self.test_results["test_start_time"])
        ).total_seconds()
        self.test_results["rate_limit_info"]["total_test_duration"] = total_duration
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Test results saved to: {filename}")
        print(f"📊 Total test duration: {total_duration/60:.1f} minutes")
        
        return filename
    
    async def run_complete_test(self) -> Dict[str, Any]:
        """
        Run the complete semantic memory integration test.
        
        Returns:
            Dict: Complete test results
        """
        print("🧪 Starting Comprehensive Semantic Memory Integration Test")
        print("=" * 60)
        
        try:
            # Step 1: Setup conversation with history
            await self.setup_conversation_with_history()
            
            # Step 2: Test semantic search retrieval
            self.test_semantic_search_retrieval()
            
            # Step 3: Test AI response with context
            await self.test_ai_response_with_context()
            
            # Step 4: Verify integration
            verification = self.verify_semantic_memory_integration()
            
            # Step 5: Save results
            results_file = self.save_test_results()
            
            print("\n" + "=" * 60)
            print("🎉 SEMANTIC MEMORY INTEGRATION TEST COMPLETE!")
            print("=" * 60)
            
            return {
                "success": verification["overall_success"],
                "results_file": results_file,
                "verification": verification,
                "conversation_id": self.conversation_id
            }
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            # Still save partial results
            self.save_test_results(f"semantic_memory_test_FAILED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            raise


async def main():
    """Main test execution function."""
    print("🤖 AI Assistant - Semantic Memory Integration Test")
    print("=" * 60)
    print("This test will:")
    print("1. Create 35+ conversation messages")
    print("2. Test semantic search retrieval")
    print("3. Verify AI responses use retrieved context")
    print("4. Save detailed test results")
    print("\n⚠️  Note: This test respects API rate limits and will take several minutes")
    print("=" * 60)
    
    # Initialize and run test
    tester = SemanticMemoryTester()
    results = await tester.run_complete_test()
    
    print(f"\n🎯 Test Result: {'SUCCESS ✅' if results['success'] else 'FAILED ❌'}")
    print(f"📄 Results saved to: {results['results_file']}")
    print(f"🔗 Conversation ID: {results['conversation_id']}")


if __name__ == "__main__":
    """Run the semantic memory integration test."""
    asyncio.run(main())
